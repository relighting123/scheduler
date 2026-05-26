import os
import pandas as pd
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from biz.services.rl.benchmark_report import (
    build_scenario_detail_df,
    capture_initial_allocation,
)
from biz.services.rl.env.scheduler_env import SchedulerEnv
from biz.services.rl.env.env_schema import load_env_schema
from biz.services.rl.inference_outputs import (
    build_allocation_pivot_df,
    build_final_allocation_df,
    build_last_process_achievement_df,
    save_action_results,
    save_inference_summary,
    save_production_logs,
)
from biz.services.rl.rts_output import (
    build_rts_rslt_mas_rows,
    format_timekey_14,
    resolve_simulation_base_datetime,
    resolve_simulation_time_range,
    save_rts_rslt_mas,
    segment_slot_time_range,
    step_to_slot_tm,
)
from biz.services.rl.training import BenchmarkTrainer
class RLSchedulerService:
    # 학습(Input) 데이터 스냅샷 식별자 (YYYYMMDDHHMMSS)
    DEFAULT_RULE_TIMEKEY = '20251020070000'

    def __init__(self, db_manager):
        # db_manager는 Core에서 주입받는다고 가정
        self.db = db_manager
        self._active_env_schema = None

    def _make_scheduler_env(self, data, max_steps=24, guidance_target_allocation=None):
        """Fixed product/process/model schema environment."""
        schema = self._active_env_schema or load_env_schema()
        schema_kwargs = {}
        if schema:
            schema_kwargs = {
                "fixed_products": schema.get("products"),
                "fixed_processes": schema.get("processes"),
                "fixed_models": schema.get("models"),
                "max_prods": schema.get("max_prods", len(schema.get("products", []))),
                "max_procs": schema.get("max_procs", len(schema.get("processes", []))),
                "max_models": schema.get("max_models", len(schema.get("models", []))),
            }
            schema_kwargs = {k: v for k, v in schema_kwargs.items() if v}
        else:
            schema_kwargs = {
                "max_prods": SchedulerEnv.DEFAULT_MAX_PRODS,
                "max_procs": SchedulerEnv.DEFAULT_MAX_PROCS,
                "max_models": SchedulerEnv.DEFAULT_MAX_MODELS,
            }
        return SchedulerEnv(
            data=data,
            max_steps=max_steps,
            guidance_target_allocation=guidance_target_allocation,
            **schema_kwargs,
        )

    def _predict_action(self, model, obs):
        """PPO predict — 관측 차원 검증."""
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        expected = model.observation_space.shape
        if obs.shape != expected:
            raise ValueError(
                f"관측 차원 불일치: env={obs.shape}, model={expected}. "
                "동일 패딩(10×10) env로 재학습하세요."
            )
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    def _resolve_rule_timekey(self, rule_timekey=None):
        """조회·학습·추론에 사용할 RULE_TIMEKEY 결정 (미지정 시 DB 최신 또는 기본값)."""
        if rule_timekey and str(rule_timekey) not in ('', 'N/A'):
            return str(rule_timekey)
        try:
            row = self.db.select_one(
                "SELECT MAX(RULE_TIMEKEY) AS RULE_TIMEKEY FROM WIP_INFO"
            )
            if row and row.get('RULE_TIMEKEY'):
                return str(row['RULE_TIMEKEY'])
        except Exception:
            pass
        return self.DEFAULT_RULE_TIMEKEY

    def _filter_data_by_rule_timekey(self, data, rule_timekey):
        """각 학습 데이터프레임을 지정 RULE_TIMEKEY 스냅샷으로 필터링."""
        filtered = {}
        for key, df in data.items():
            if df is None or df.empty:
                filtered[key] = df
                continue
            if 'RULE_TIMEKEY' not in df.columns:
                filtered[key] = df
                continue
            snapshot = df[df['RULE_TIMEKEY'] == rule_timekey].copy()
            filtered[key] = snapshot.drop(columns=['RULE_TIMEKEY'], errors='ignore')
        return filtered

    def _normalize_timekey_arg(self, value):
        if value is None:
            return None
        s = str(value).strip()
        if s in ('', 'N/A'):
            return None
        return s

    def _list_rule_timekeys_in_range(self, from_rule_timekey=None, to_rule_timekey=None):
        """학습 구간 [from, to] 내 DISTINCT RULE_TIMEKEY 목록 (문자열 정렬 기준)."""
        from_tk = self._normalize_timekey_arg(from_rule_timekey)
        to_tk = self._normalize_timekey_arg(to_rule_timekey)

        if not from_tk and not to_tk:
            return [self._resolve_rule_timekey(None)]
        if from_tk and not to_tk:
            to_tk = from_tk
        elif to_tk and not from_tk:
            from_tk = to_tk

        try:
            rows = self.db.select_list(
                "SELECT DISTINCT RULE_TIMEKEY FROM WIP_INFO ORDER BY RULE_TIMEKEY"
            )
            keys = [
                str(r['RULE_TIMEKEY'])
                for r in rows
                if r.get('RULE_TIMEKEY') and from_tk <= str(r['RULE_TIMEKEY']) <= to_tk
            ]
            if keys:
                return keys
        except Exception:
            pass
        return [self._resolve_rule_timekey(to_tk)]

    def fetch_training_snapshots(
        self,
        from_rule_timekey=None,
        to_rule_timekey=None,
        rule_timekey=None,
    ):
        """학습에 사용할 스냅샷 목록. 단일 rule_timekey 또는 from~to 구간."""
        single = self._normalize_timekey_arg(rule_timekey)
        if single:
            return [self.fetch_data(rule_timekey=single)]
        keys = self._list_rule_timekeys_in_range(from_rule_timekey, to_rule_timekey)
        snapshots = [self.fetch_data(rule_timekey=k) for k in keys]
        print(
            f"[학습 데이터] RULE_TIMEKEY {len(snapshots)}개 스냅샷 "
            f"({', '.join(keys)})"
        )
        return snapshots

    def _create_learning_tables(self):
        """학습(Input) 테이블 7종 생성 (RULE_TIMEKEY 포함)."""
        self.db.execute("""
            CREATE TABLE WIP_INFO (
                RULE_TIMEKEY VARCHAR2(14), PLAN_PROD_KEY VARCHAR2(50),
                OPER_ID VARCHAR2(50), OPER_SEQ NUMBER, WIP_QTY NUMBER
            )
        """)
        self.db.execute("""
            CREATE TABLE UPH_INFO (
                RULE_TIMEKEY VARCHAR2(14), PLAN_PROD_KEY VARCHAR2(50),
                OPER_ID VARCHAR2(50), EQP_MODEL_CD VARCHAR2(50), UPH NUMBER
            )
        """)
        self.db.execute("""
            CREATE TABLE EQP_QTY_INFO (
                RULE_TIMEKEY VARCHAR2(14), BATCH_ID VARCHAR2(50),
                EQP_MODEL_CD VARCHAR2(50), TIME_SLOT VARCHAR2(50), EQP_QTY NUMBER
            )
        """)
        self.db.execute("""
            CREATE TABLE AVAIL_INFO (
                RULE_TIMEKEY VARCHAR2(14), PLAN_PROD_KEY VARCHAR2(50),
                OPER_ID VARCHAR2(50), EQP_MODEL_CD VARCHAR2(50), AVAIL_YN VARCHAR2(10)
            )
        """)
        self.db.execute("""
            CREATE TABLE BATCH_TOOL_INFO (
                RULE_TIMEKEY VARCHAR2(14), BATCH_ID VARCHAR2(50),
                PLAN_PROD_KEY VARCHAR2(50), OPER_ID VARCHAR2(50)
            )
        """)
        self.db.execute("""
            CREATE TABLE TOOL_QTY_INFO (
                RULE_TIMEKEY VARCHAR2(14), BATCH_ID VARCHAR2(50),
                EQP_MODEL_CD VARCHAR2(50), TOOL_QTY NUMBER
            )
        """)
        self.db.execute("""
            CREATE TABLE PLAN_INFO (
                RULE_TIMEKEY VARCHAR2(14), PLAN_PROD_KEY VARCHAR2(50),
                OPER_ID VARCHAR2(50), START_TIME VARCHAR2(50),
                END_TIME VARCHAR2(50), PLAN_QTY NUMBER
            )
        """)

    def _create_output_tables(self):
        """추론(Output) 테이블 생성."""
        self.db.execute("""
            CREATE TABLE RTD_CONV_INF (
                RULE_TIMEKEY VARCHAR2(20),
                FROM_BATCH VARCHAR2(50),
                FROM_PLAN_PROD_KEY VARCHAR2(50),
                FROM_OPER_ID VARCHAR2(50),
                EQP_MODEL_CD VARCHAR2(50),
                TO_BATCH_ID VARCHAR2(50),
                TO_PLAN_PROD_KEY VARCHAR2(50),
                TO_OPER_ID VARCHAR2(50),
                START_CONV_TIME VARCHAR2(20),
                EQP_QTY NUMBER
            )
        """)
        self.db.execute("""
            CREATE TABLE RTS_RSLT_MAS (
                RULE_TIMEKEY VARCHAR2(50) NOT NULL,
                SEQ_NO NUMBER NOT NULL,
                EQP_ID VARCHAR2(50) NOT NULL,
                EQP_MODEL_CD VARCHAR2(50),
                BATCH_ID VARCHAR2(50),
                START_TM VARCHAR2(14),
                END_TM VARCHAR2(14),
                PLAN_PROD_ATTR_VAL VARCHAR2(200),
                PROD_QTY VARCHAR2(50),
                CUM_PROD_QTY VARCHAR2(50),
                CRT_USET_ID VARCHAR2(50),
                CRT_TM VARCHAR2(14),
                CONSTRAINT PK_RTS_RSLT_MAS PRIMARY KEY (RULE_TIMEKEY, EQP_ID, SEQ_NO)
            )
        """)

    def _format_timekey_14(self, time_value):
        return format_timekey_14(time_value)

    def _resolve_simulation_time_range(self, data):
        return resolve_simulation_time_range(data)

    def _resolve_simulation_base_datetime(self, data, rule_timekey=None):
        return resolve_simulation_base_datetime(data, rule_timekey=rule_timekey)

    def _step_to_slot_tm(self, base_dt, step_index: int) -> str:
        return step_to_slot_tm(base_dt, step_index)

    def _segment_slot_time_range(self, data, seg, max_steps: int, rule_timekey=None):
        return segment_slot_time_range(data, seg, max_steps, rule_timekey=rule_timekey)

    def _build_rts_rslt_mas_rows(self, env, data, rule_timekey, crt_user_id='SYSTEM'):
        return build_rts_rslt_mas_rows(env, data, rule_timekey, crt_user_id=crt_user_id)

    def save_rts_rslt_mas(self, env, data, rule_timekey, crt_user_id='SYSTEM'):
        return save_rts_rslt_mas(
            self.db,
            env,
            data,
            rule_timekey,
            crt_user_id=crt_user_id,
        )

    def init_db_scenario(self):
        """휴리스틱 함정(Trap) 시나리오용 DB 초기화 (DROP -> CREATE -> INSERT)"""
        print("\n[DB 설정] DB 시나리오 초기화를 시작합니다 (Heuristic Trap Scenario)...")
        tables = [
            "WIP_INFO", "UPH_INFO", "EQP_QTY_INFO", "AVAIL_INFO",
            "BATCH_TOOL_INFO", "TOOL_QTY_INFO", "PLAN_INFO", "RTD_CONV_INF", "RTS_RSLT_MAS"
        ]

        # 1. DROP Tables
        for table in tables:
            try:
                self.db.execute(f"DROP TABLE {table}")
            except Exception:
                pass # 테이블이 없으면 무시

        # 2. CREATE Tables
        self._create_learning_tables()
        self._create_output_tables()
        tk = self.DEFAULT_RULE_TIMEKEY

        # 3. INSERT Data (Trap Scenario)
        # WIP: OP10 = 5000 (Trap trigger), OP20 = 500
        self.db.execute(f"INSERT INTO WIP_INFO VALUES ('{tk}', 'P1', 'OP10', 10, 5000)")
        self.db.execute(f"INSERT INTO WIP_INFO VALUES ('{tk}', 'P1', 'OP20', 20, 500)")
        
        # UPH: 100/hr for both operations
        self.db.execute(f"INSERT INTO UPH_INFO VALUES ('{tk}', 'P1', 'OP10', 'MODEL_A', 100)")
        self.db.execute(f"INSERT INTO UPH_INFO VALUES ('{tk}', 'P1', 'OP20', 'MODEL_A', 100)")
        
        # EQP_QTY (Initial allocation): 5 units perfectly balanced
        self.db.execute(f"INSERT INTO EQP_QTY_INFO VALUES ('{tk}', 'B1', 'MODEL_A', '2026051800', 5)")
        self.db.execute(f"INSERT INTO EQP_QTY_INFO VALUES ('{tk}', 'B2', 'MODEL_A', '2026051800', 5)")
        
        # AVAIL
        self.db.execute(f"INSERT INTO AVAIL_INFO VALUES ('{tk}', 'P1', 'OP10', 'MODEL_A', 'Y')")
        self.db.execute(f"INSERT INTO AVAIL_INFO VALUES ('{tk}', 'P1', 'OP20', 'MODEL_A', 'Y')")
        
        # BATCH
        self.db.execute(f"INSERT INTO BATCH_TOOL_INFO VALUES ('{tk}', 'B1', 'P1', 'OP10')")
        self.db.execute(f"INSERT INTO BATCH_TOOL_INFO VALUES ('{tk}', 'B2', 'P1', 'OP20')")
        
        # PLAN
        self.db.execute(f"INSERT INTO PLAN_INFO VALUES ('{tk}', 'P1', 'OP10', '2026051800', '2026051824', 4000)")
        self.db.execute(f"INSERT INTO PLAN_INFO VALUES ('{tk}', 'P1', 'OP20', '2026051800', '2026051824', 4000)")
        
        print("[성공] DB 시나리오 초기화가 완료되었습니다 (WIP_INFO 등 7개 테이블).")

    def fetch_data(self, rule_timekey=None):
        """DB에서 학습 데이터를 조회합니다. rule_timekey로 스냅샷(기간)을 지정합니다."""
        resolved_tk = self._resolve_rule_timekey(rule_timekey)
        try:
            wip_data = pd.DataFrame(
                self.db.select_list(
                    "SELECT RULE_TIMEKEY, PLAN_PROD_KEY, OPER_ID, OPER_SEQ, WIP_QTY FROM WIP_INFO"
                ),
                columns=['RULE_TIMEKEY', 'PLAN_PROD_KEY', 'OPER_ID', 'OPER_SEQ', 'WIP_QTY'],
            )
            uph_data = pd.DataFrame(
                self.db.select_list(
                    "SELECT RULE_TIMEKEY, PLAN_PROD_KEY, OPER_ID, EQP_MODEL_CD, UPH FROM UPH_INFO"
                ),
                columns=['RULE_TIMEKEY', 'PLAN_PROD_KEY', 'OPER_ID', 'EQP_MODEL_CD', 'UPH'],
            )
            eqp_qty_data = pd.DataFrame(
                self.db.select_list(
                    "SELECT RULE_TIMEKEY, BATCH_ID, EQP_MODEL_CD, TIME_SLOT, EQP_QTY FROM EQP_QTY_INFO"
                ),
                columns=['RULE_TIMEKEY', 'BATCH_ID', 'EQP_MODEL_CD', 'TIME_SLOT', 'EQP_QTY'],
            )
            avail_data = pd.DataFrame(
                self.db.select_list(
                    "SELECT RULE_TIMEKEY, PLAN_PROD_KEY, OPER_ID, EQP_MODEL_CD, AVAIL_YN FROM AVAIL_INFO"
                ),
                columns=['RULE_TIMEKEY', 'PLAN_PROD_KEY', 'OPER_ID', 'EQP_MODEL_CD', 'AVAIL_YN'],
            )
            batch_tool_data = pd.DataFrame(
                self.db.select_list(
                    "SELECT RULE_TIMEKEY, BATCH_ID, PLAN_PROD_KEY, OPER_ID FROM BATCH_TOOL_INFO"
                ),
                columns=['RULE_TIMEKEY', 'BATCH_ID', 'PLAN_PROD_KEY', 'OPER_ID'],
            )
            tool_qty_data = pd.DataFrame(
                self.db.select_list(
                    "SELECT RULE_TIMEKEY, BATCH_ID, EQP_MODEL_CD, TOOL_QTY FROM TOOL_QTY_INFO"
                ),
                columns=['RULE_TIMEKEY', 'BATCH_ID', 'EQP_MODEL_CD', 'TOOL_QTY'],
            )
            plan_data = pd.DataFrame(
                self.db.select_list(
                    "SELECT RULE_TIMEKEY, PLAN_PROD_KEY, OPER_ID, START_TIME, END_TIME, PLAN_QTY FROM PLAN_INFO"
                ),
                columns=['RULE_TIMEKEY', 'PLAN_PROD_KEY', 'OPER_ID', 'START_TIME', 'END_TIME', 'PLAN_QTY'],
            )

            if wip_data.empty:
                wip_data = pd.DataFrame(columns=['RULE_TIMEKEY', 'PLAN_PROD_KEY', 'OPER_ID', 'OPER_SEQ', 'WIP_QTY'])
            if uph_data.empty:
                uph_data = pd.DataFrame(columns=['RULE_TIMEKEY', 'PLAN_PROD_KEY', 'OPER_ID', 'EQP_MODEL_CD', 'UPH'])
            if eqp_qty_data.empty:
                eqp_qty_data = pd.DataFrame(columns=['RULE_TIMEKEY', 'BATCH_ID', 'EQP_MODEL_CD', 'TIME_SLOT', 'EQP_QTY'])
            if avail_data.empty:
                avail_data = pd.DataFrame(columns=['RULE_TIMEKEY', 'PLAN_PROD_KEY', 'OPER_ID', 'EQP_MODEL_CD', 'AVAIL_YN'])
            if batch_tool_data.empty:
                batch_tool_data = pd.DataFrame(columns=['RULE_TIMEKEY', 'BATCH_ID', 'PLAN_PROD_KEY', 'OPER_ID'])
            if tool_qty_data.empty:
                tool_qty_data = pd.DataFrame(columns=['RULE_TIMEKEY', 'BATCH_ID', 'EQP_MODEL_CD', 'TOOL_QTY'])
            if plan_data.empty:
                plan_data = pd.DataFrame(columns=['RULE_TIMEKEY', 'PLAN_PROD_KEY', 'OPER_ID', 'START_TIME', 'END_TIME', 'PLAN_QTY'])

            raw = {
                'wip_info': wip_data,
                'uph_info': uph_data,
                'eqp_qty_info': eqp_qty_data,
                'avail_info': avail_data,
                'batch_tool_info': batch_tool_data,
                'tool_qty_info': tool_qty_data,
                'plan_info': plan_data,
            }
            data = self._filter_data_by_rule_timekey(raw, resolved_tk)
            print(f"[성공] DB에서 RULE_TIMEKEY={resolved_tk} 스냅샷 데이터를 조회했습니다.")
            return data

        except Exception as e:
            print(f"[경고] DB 연동 실패 (또는 테이블 없음): {e}")
            print("데이터를 조회할 수 없습니다. DB 초기화(init_db_scenario)가 올바르게 수행되었는지 확인하세요.")
            return self._filter_data_by_rule_timekey({
                'wip_info': pd.DataFrame(columns=['RULE_TIMEKEY', 'PLAN_PROD_KEY', 'OPER_ID', 'OPER_SEQ', 'WIP_QTY']),
                'uph_info': pd.DataFrame(columns=['RULE_TIMEKEY', 'PLAN_PROD_KEY', 'OPER_ID', 'EQP_MODEL_CD', 'UPH']),
                'eqp_qty_info': pd.DataFrame(columns=['RULE_TIMEKEY', 'BATCH_ID', 'EQP_MODEL_CD', 'TIME_SLOT', 'EQP_QTY']),
                'avail_info': pd.DataFrame(columns=['RULE_TIMEKEY', 'PLAN_PROD_KEY', 'OPER_ID', 'EQP_MODEL_CD', 'AVAIL_YN']),
                'batch_tool_info': pd.DataFrame(columns=['RULE_TIMEKEY', 'BATCH_ID', 'PLAN_PROD_KEY', 'OPER_ID']),
                'tool_qty_info': pd.DataFrame(columns=['RULE_TIMEKEY', 'BATCH_ID', 'EQP_MODEL_CD', 'TOOL_QTY']),
                'plan_info': pd.DataFrame(columns=['RULE_TIMEKEY', 'PLAN_PROD_KEY', 'OPER_ID', 'START_TIME', 'END_TIME', 'PLAN_QTY']),
            }, resolved_tk)

    def generate_expert_data(
        self,
        env,
        num_samples=5000,
        expert_cls=None,
        target_allocation=None,
    ):
        return BenchmarkTrainer(self).generate_expert_data(
            env,
            num_samples=num_samples,
            expert_cls=expert_cls,
            target_allocation=target_allocation,
        )

    def _score_benchmark_model(self, model, scenario_payloads, max_steps=24):
        return BenchmarkTrainer(self).score_benchmark_model(
            model,
            scenario_payloads,
            max_steps=max_steps,
        )

    def pretrain_behavior_cloning(
        self,
        model,
        expert_obs,
        expert_actions,
        epochs=20,
        batch_size=64,
        learning_rate=3e-4,
    ):
        return BenchmarkTrainer(self).pretrain_behavior_cloning(
            model,
            expert_obs,
            expert_actions,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

    def _collect_op20_metrics(self, env, products=None):
        """최종 공정(OP20) 기준 제품별·평균 계획달성률 및 장비 전환 횟수 집계."""
        if products is None:
            products = ['P1', 'P2', 'P3']
        results = {}
        for p_idx, p_name in enumerate(env.products):
            if p_name not in products:
                continue
            s_idx = env.proc_idx.get('OP20')
            if s_idx is None:
                continue
            prod_qty = env.produced[p_idx, s_idx]
            plan_qty = env.plan[p_idx, s_idx]
            rate = (prod_qty / plan_qty * 100.0) if plan_qty > 0 else 0.0
            results[f"{p_name}_OP20_ACHIEVEMENT"] = round(rate, 2)
        avg = round(sum(results.values()) / len(results), 2) if results else 0.0
        results['AVG_ACHIEVEMENT'] = avg
        return results

    def _run_simulation_with_policy(
        self,
        data,
        max_steps=24,
        model=None,
        expert_cls=None,
        method_name='simulation',
        model_path='scheduler_ppo_model',
        target_allocation=None,
    ):
        """단일 데이터 스냅샷으로 24스텝 시뮬레이션 후 지표·전환 횟수 반환."""
        env = self._make_scheduler_env(data, max_steps=max_steps)
        if expert_cls is not None:
            if target_allocation is not None and expert_cls.__name__ == 'OptimalExpert':
                expert = expert_cls(env, target_allocation=target_allocation)
            else:
                expert = expert_cls(env)
        else:
            expert = None

        obs, _ = env.reset()
        initial_eqp = capture_initial_allocation(env)
        transfers = 0
        done = False
        while not done:
            if expert is not None:
                action = expert.select_action()
            elif model is not None:
                try:
                    action = self._predict_action(model, obs)
                except ValueError as exc:
                    print(f"[경고] 모델/env 차원 불일치 - 무작위 액션으로 전환합니다. {exc}")
                    model = None
                    action = env.action_space.sample()
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get('transfers'):
                transfers += len(info['transfers'])
            done = terminated or truncated

        env.print_final_summary(method_name=method_name, save_excel=False)
        metrics = self._collect_op20_metrics(env)
        metrics['TRANSFERS'] = transfers
        detail_df, avg_util = build_scenario_detail_df(env, initial_eqp, method_name)
        metrics['AVG_UTILIZATION'] = avg_util
        return metrics, env, detail_df

    def _write_benchmark_excel_sheets(
        self, writer, scenario_id, comparison_df, detail_by_method, optimal_ref
    ):
        prefix = scenario_id[:20]
        comparison_df.to_excel(writer, sheet_name=f'{prefix}_COMP', index=False)
        pd.DataFrame([optimal_ref]).to_excel(writer, sheet_name=f'{prefix}_REF', index=False)
        sheet_names = {
            '1. 정답지 (Optimal GT)': f'{prefix}_OPT',
            '2. 휴리스틱 (Expert)': f'{prefix}_HEU',
            '3. 강화학습 (RL PPO)': f'{prefix}_RL',
        }
        for method_label, detail_df in detail_by_method.items():
            sheet = sheet_names.get(method_label, f'{prefix}_DET')[:31]
            detail_df.to_excel(writer, sheet_name=sheet, index=False)

    def evaluate_on_benchmark_dataset(
        self,
        benchmark_dataset='benchmark_dataset',
        model_path='scheduler_ppo_model',
        max_steps=24,
        save_excel=True,
        excel_writer=None,
    ):
        """test/data 벤치마크: 초기·최종 장비대수, 생산/계획/달성률, 평균 가동률."""
        from biz.services.rl.benchmark_evaluator import BenchmarkEvaluator
        return BenchmarkEvaluator(self).evaluate_dataset(
            benchmark_dataset=benchmark_dataset,
            model_path=model_path,
            max_steps=max_steps,
            save_excel=save_excel,
            excel_writer=excel_writer,
        )


    def evaluate_all_benchmark_datasets(self, datasets=None, model_path='scheduler_ppo_model', max_steps=24):
        from biz.services.rl.test_data_loader import TestDataLoader
        loader = TestDataLoader()
        datasets = datasets or loader.list_scenarios() or ['benchmark_dataset']
        print("\n" + "#" * 80)
        print(f" [전체 벤치마크 평가 - {len(datasets)}개: {', '.join(datasets)}]")
        print("#" * 80)
        results = {}
        log_dir = os.path.join(os.getcwd(), 'logs', 'simulation_logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unified_path = os.path.join(log_dir, f'benchmark_unified_report_{timestamp}.xlsx')
        with pd.ExcelWriter(unified_path) as writer:
            summary_rows = []
            for scenario in datasets:
                report = self.evaluate_on_benchmark_dataset(
                    benchmark_dataset=scenario, model_path=model_path, max_steps=max_steps,
                    save_excel=False, excel_writer=writer,
                )
                results[scenario] = report
                comp = report['comparison_df'].copy()
                comp.insert(0, '데이터셋', scenario)
                summary_rows.append(comp)
            if summary_rows:
                pd.concat(summary_rows, ignore_index=True).to_excel(writer, sheet_name='ALL_SUMMARY', index=False)
        print(f"\n[성공] 통합 벤치마크 리포트: {unified_path}")
        return results

    def train_model(
        self,
        total_timesteps=10000,
        pretrain_bc=True,
        n_envs=4,
        batch_size=64,
        n_steps=1024,
        bc_samples=5000,
        bc_epochs=20,
        ent_coef=0.05,
        rule_timekey=None,
        from_rule_timekey=None,
        to_rule_timekey=None,
        run_test_eval=True,
        benchmark_dataset="benchmark_dataset",
        benchmark_datasets=None,
        evaluate_all_benchmarks=True,
    ):
        return BenchmarkTrainer(self).train_model(
            total_timesteps=total_timesteps,
            pretrain_bc=pretrain_bc,
            n_envs=n_envs,
            batch_size=batch_size,
            n_steps=n_steps,
            bc_samples=bc_samples,
            bc_epochs=bc_epochs,
            ent_coef=ent_coef,
            rule_timekey=rule_timekey,
            from_rule_timekey=from_rule_timekey,
            to_rule_timekey=to_rule_timekey,
            run_test_eval=run_test_eval,
            benchmark_dataset=benchmark_dataset,
            benchmark_datasets=benchmark_datasets,
            evaluate_all_benchmarks=evaluate_all_benchmarks,
        )

    def _build_final_allocation_df(self, env):
        return build_final_allocation_df(env)

    def _build_last_process_achievement_df(self, env, data):
        return build_last_process_achievement_df(env, data)

    def _build_allocation_pivot_df(self, allocation_df):
        return build_allocation_pivot_df(allocation_df)

    def save_inference_summary(
        self,
        rule_timekey,
        allocation_df,
        achievement_df,
        file_prefix="inference_summary",
    ):
        return save_inference_summary(
            rule_timekey,
            allocation_df,
            achievement_df,
            file_prefix=file_prefix,
        )

    def run_inference(self, rule_timekey=None):
        """추론 수행.

        - 조회·출력(RTD_CONV 등) RULE_TIMEKEY: rule_timekey 지정, 미지정 시 WIP_INFO MAX(RULE_TIMEKEY)
        """
        resolved_timekey = self._resolve_rule_timekey(rule_timekey)
        print(f"[추론] RULE_TIMEKEY={resolved_timekey} (입력 스냅샷·결과 출력 공통)")
        data = self.fetch_data(rule_timekey=resolved_timekey)
        model_path = "scheduler_ppo_model"
        model = None
        try:
            model = PPO.load(model_path)
        except Exception:
            print("저장된 모델이 없습니다. 임의의 액션으로 시뮬레이션 합니다.")

        env = self._make_scheduler_env(data)
        if model is not None:
            print(
                f"[추론 env] obs={env.observation_space.shape}, "
                f"model obs={model.observation_space.shape}"
            )

        obs, _ = env.reset()
        done = False

        results = []

        while not done:
            if model:
                try:
                    action = self._predict_action(model, obs)
                except ValueError as exc:
                    print(f"[경고] 모델/env 차원 불일치 - 무작위 액션으로 전환합니다. {exc}")
                    model = None
                    action = env.action_space.sample()
            else:
                action = env.action_space.sample()
            
            # Action logic uses RTS style: target selection. The env handles the transfers
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            
            # Capture the transfer logs generated by the step
            if 'transfers' in info:
                for transfer in info['transfers']:
                    results.append({
                        'RULE_TIMEKEY': resolved_timekey,
                        'FROM_PLAN_PROD_KEY': transfer['FROM_PROD'],
                        'FROM_OPER_ID': transfer['FROM_PROC'],
                        'EQP_MODEL_CD': transfer['MODEL'],
                        'TO_PLAN_PROD_KEY': transfer['TO_PROD'],
                        'TO_OPER_ID': transfer['TO_PROC'],
                        'START_CONV_TIME': datetime.now().strftime("%Y%m%d%H%M%S"),
                        'EQP_QTY': 1
                    })
            
        # 최종 실적 및 장비 가동 요약 출력
        env.print_final_summary(method_name="inference")
        self.save_results(results)
        if hasattr(env, 'production_logs') and env.production_logs:
            self.save_production_logs(env.production_logs)

        allocation_df = self._build_final_allocation_df(env)
        achievement_df = self._build_last_process_achievement_df(env, data)
        self.save_inference_summary(resolved_timekey, allocation_df, achievement_df)
        self.save_rts_rslt_mas(env, data, resolved_timekey)

        return results

    def save_production_logs(self, logs):
        return save_production_logs(logs)

    def save_results(self, results):
        return save_action_results(results)

    def run_benchmark_evaluation(self, datasets=None, model_path='scheduler_ppo_model', max_steps=24):
        """벤치마크 CSV 데이터셋으로 저장된 모델을 validation/evaluation만 수행."""
        print("\n" + "=" * 80)
        print(" [벤치마크 데이터셋 검증 (Optimal vs Heuristic vs RL)]")
        print("=" * 80)
        return self.evaluate_all_benchmark_datasets(
            datasets=datasets,
            model_path=model_path,
            max_steps=max_steps,
        )
