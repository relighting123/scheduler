import os
import re
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from biz.services.rl.callbacks import PlottingCallback
from biz.services.rl.env.scheduler_env import SchedulerEnv
from biz.services.rl.env.snapshot_rotation_env import SnapshotRotationEnv


class RLSchedulerService:
    # 학습(Input) 데이터 스냅샷 식별자 (YYYYMMDDHHMMSS)
    DEFAULT_RULE_TIMEKEY = '20251020070000'

    def __init__(self, db_manager):
        # db_manager는 Core에서 주입받는다고 가정
        self.db = db_manager

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
                CONSTRAINT PK_RTS_RSLT_MAS PRIMARY KEY (RULE_TIMEKEY, EQP_ID)
            )
        """)

    def _format_timekey_14(self, time_value):
        """PLAN 시간값을 YYYYMMDDHHMMSS(14자리)로 정규화."""
        return str(time_value).strip().ljust(14, '0')[:14]

    def _resolve_simulation_time_range(self, data):
        """시뮬레이션 구간 START_TM / END_TM (PLAN_INFO 기준)."""
        plan_df = data.get('plan_info', pd.DataFrame())
        if plan_df.empty or not {'START_TIME', 'END_TIME'}.issubset(plan_df.columns):
            now = datetime.now().strftime("%Y%m%d%H%M%S")
            return now, now
        start_tm = self._format_timekey_14(plan_df['START_TIME'].astype(str).min())
        end_tm = self._format_timekey_14(plan_df['END_TIME'].astype(str).max())
        return start_tm, end_tm

    def _build_rts_rslt_mas_rows(self, env, data, rule_timekey, crt_user_id='SYSTEM'):
        """시뮬레이션 최종 상태를 RTS_RSLT_MAS Output 행으로 변환 (설비 풀 기반 EQP_ID)."""
        start_tm, end_tm = self._resolve_simulation_time_range(data)
        crt_tm = datetime.now().strftime("%Y%m%d%H%M%S")
        rows = []

        deployed_units = env.get_deployed_equipment_units() if hasattr(env, 'get_deployed_equipment_units') else []
        if deployed_units:
            for unit in deployed_units:
                prod_qty_str = str(round(float(unit.produced_qty), 4))
                rows.append({
                    'RULE_TIMEKEY': str(rule_timekey),
                    'EQP_ID': unit.eqp_id,
                    'EQP_MODEL_CD': unit.eqp_model_cd,
                    'BATCH_ID': unit.batch_id,
                    'START_TM': start_tm,
                    'END_TM': end_tm,
                    'PLAN_PROD_ATTR_VAL': unit.plan_prod_attr_val or '',
                    'PROD_QTY': prod_qty_str,
                    'CUM_PROD_QTY': prod_qty_str,
                    'CRT_USET_ID': crt_user_id,
                    'CRT_TM': crt_tm,
                })
            return rows

        # fallback: equipment registry 미사용 환경
        for p_idx, prod in enumerate(env.products):
            if prod.startswith("PAD_PROD_"):
                continue
            for s_idx, oper in enumerate(env.processes):
                if oper.startswith("PAD_PROC_"):
                    continue
                batch_id = env.batch_id_map.get((prod, oper), 'EQP')
                produced_qty = float(env.produced[p_idx, s_idx])
                for m_idx, model in enumerate(env.models):
                    eqp_count = int(round(float(env.active_eqp[p_idx, s_idx, m_idx])))
                    if eqp_count <= 0:
                        continue
                    per_eqp_prod = produced_qty / eqp_count if eqp_count > 0 else 0.0
                    prod_qty_str = str(round(per_eqp_prod, 4))
                    for seq in range(1, eqp_count + 1):
                        rows.append({
                            'RULE_TIMEKEY': str(rule_timekey),
                            'EQP_ID': f"{model}-{seq:05d}",
                            'EQP_MODEL_CD': model,
                            'BATCH_ID': batch_id,
                            'START_TM': start_tm,
                            'END_TM': end_tm,
                            'PLAN_PROD_ATTR_VAL': f"{prod}|{oper}",
                            'PROD_QTY': prod_qty_str,
                            'CUM_PROD_QTY': prod_qty_str,
                            'CRT_USET_ID': crt_user_id,
                            'CRT_TM': crt_tm,
                        })
        return rows

    def save_rts_rslt_mas(self, env, data, rule_timekey, crt_user_id='SYSTEM'):
        """RTS_RSLT_MAS Output 저장 (동일 RULE_TIMEKEY 삭제 후 INSERT)."""
        rows = self._build_rts_rslt_mas_rows(env, data, rule_timekey, crt_user_id=crt_user_id)
        if not rows:
            print("\n[RTS_RSLT_MAS] 저장할 결과가 없습니다.")
            return rows

        df = pd.DataFrame(rows)
        print(f"\n[RTS_RSLT_MAS] 총 {len(df)}건 Output")
        print("-" * 80)
        print(df.to_string(index=False))
        print("-" * 80)

        log_dir = os.path.join(os.getcwd(), 'logs', 'simulation_logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_timekey = str(rule_timekey).replace(" ", "_")
        file_path = os.path.join(log_dir, f'rts_rslt_mas_{safe_timekey}_{timestamp}.xlsx')
        df.to_excel(file_path, index=False)
        print(f"[성공] RTS_RSLT_MAS Output이 엑셀로 저장되었습니다: {file_path}")

        if self.db is None:
            return rows

        try:
            self.db.execute(
                "DELETE FROM RTS_RSLT_MAS WHERE RULE_TIMEKEY = :tk",
                {"tk": str(rule_timekey)},
            )
            self.db.bulk_execute(
                """
                INSERT INTO RTS_RSLT_MAS (
                    RULE_TIMEKEY, EQP_ID, EQP_MODEL_CD, BATCH_ID, START_TM, END_TM,
                    PLAN_PROD_ATTR_VAL, PROD_QTY, CUM_PROD_QTY, CRT_USET_ID, CRT_TM
                ) VALUES (
                    :RULE_TIMEKEY, :EQP_ID, :EQP_MODEL_CD, :BATCH_ID, :START_TM, :END_TM,
                    :PLAN_PROD_ATTR_VAL, :PROD_QTY, :CUM_PROD_QTY, :CRT_USET_ID, :CRT_TM
                )
                """,
                rows,
            )
            print(f"[성공] RTS_RSLT_MAS {len(rows)}건 DB 저장 완료 (RULE_TIMEKEY={rule_timekey})")
        except Exception as exc:
            if 'ORA-00942' in str(exc) or 'table or view does not exist' in str(exc).lower():
                print("[안내] RTS_RSLT_MAS 테이블이 없어 DB 저장을 건너뜁니다. schema/scheduler_tables.sql을 적용하세요.")
            else:
                print(f"[경고] RTS_RSLT_MAS DB 저장 실패: {exc}")
        return rows

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

    def init_benchmark_dataset_scenario(self):
        """벤치마크 데이터셋(test/data/benchmark_dataset)을 DB에 적재합니다."""
        print("\n[DB 설정] 벤치마크 데이터셋 시나리오 초기화를 시작합니다...")
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

        # 3. INSERT Data (벤치마크 데이터셋)
        # WIP: 충분한 재공 부여
        wips = [
            ('P1', 'OP10', 10, 15000), ('P1', 'OP20', 20, 2000),
            ('P2', 'OP10', 10, 10000), ('P2', 'OP20', 20, 1000),
            ('P3', 'OP10', 10, 6000),  ('P3', 'OP20', 20, 500)
        ]
        for p, s, seq, q in wips:
            self.db.execute(f"INSERT INTO WIP_INFO VALUES ('{tk}', '{p}', '{s}', {seq}, {q})")
            
        # UPH: UPH 분산 및 전용모델 특성 반영
        # MODEL_A: P1에 고효율(100), P2에 보통(80), P3 불가(0)
        # MODEL_B: P1에 저효율(70), P2에 고효율(120), P3에 저효율(60)
        # MODEL_C: P1/P2 불가(0), P3에 고효율 전용(100)
        uphs = [
            ('P1', 'OP10', 'MODEL_A', 100), ('P1', 'OP20', 'MODEL_A', 100),
            ('P1', 'OP10', 'MODEL_B', 70),  ('P1', 'OP20', 'MODEL_B', 70),
            ('P2', 'OP10', 'MODEL_A', 80),  ('P2', 'OP20', 'MODEL_A', 80),
            ('P2', 'OP10', 'MODEL_B', 120), ('P2', 'OP20', 'MODEL_B', 120),
            ('P3', 'OP10', 'MODEL_B', 60),  ('P3', 'OP20', 'MODEL_B', 60),
            ('P3', 'OP10', 'MODEL_C', 100), ('P3', 'OP20', 'MODEL_C', 100)
        ]
        for p, s, m, u in uphs:
            self.db.execute(f"INSERT INTO UPH_INFO VALUES ('{tk}', '{p}', '{s}', '{m}', {u})")
            
        # AVAIL
        avails = [
            ('P1', 'OP10', 'MODEL_A', 'Y'), ('P1', 'OP20', 'MODEL_A', 'Y'),
            ('P1', 'OP10', 'MODEL_B', 'Y'), ('P1', 'OP20', 'MODEL_B', 'Y'),
            ('P1', 'OP10', 'MODEL_C', 'N'), ('P1', 'OP20', 'MODEL_C', 'N'),
            
            ('P2', 'OP10', 'MODEL_A', 'Y'), ('P2', 'OP20', 'MODEL_A', 'Y'),
            ('P2', 'OP10', 'MODEL_B', 'Y'), ('P2', 'OP20', 'MODEL_B', 'Y'),
            ('P2', 'OP10', 'MODEL_C', 'N'), ('P2', 'OP20', 'MODEL_C', 'N'),
            
            ('P3', 'OP10', 'MODEL_A', 'N'), ('P3', 'OP20', 'MODEL_A', 'N'),
            ('P3', 'OP10', 'MODEL_B', 'Y'), ('P3', 'OP20', 'MODEL_B', 'Y'),
            ('P3', 'OP10', 'MODEL_C', 'Y'), ('P3', 'OP20', 'MODEL_C', 'Y')
        ]
        for p, s, m, a in avails:
            self.db.execute(f"INSERT INTO AVAIL_INFO VALUES ('{tk}', '{p}', '{s}', '{m}', '{a}')")
            
        # BATCH
        batches = [
            ('B1', 'P1', 'OP10'), ('B2', 'P1', 'OP20'),
            ('B3', 'P2', 'OP10'), ('B4', 'P2', 'OP20'),
            ('B5', 'P3', 'OP10'), ('B6', 'P3', 'OP20')
        ]
        for b, p, s in batches:
            self.db.execute(f"INSERT INTO BATCH_TOOL_INFO VALUES ('{tk}', '{b}', '{p}', '{s}')")
            
        # TOOL QTY
        for b in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']:
            for m in ['MODEL_A', 'MODEL_B', 'MODEL_C']:
                self.db.execute(f"INSERT INTO TOOL_QTY_INFO VALUES ('{tk}', '{b}', '{m}', 20)")
                
        # EQP QTY (초기 비효율적 할당 - 병목 유발)
        # MODEL_A (10대): B1(P1-OP10) 5대, B3(P2-OP10) 5대. (OP20 0대)
        # MODEL_B (8대): B2(P1-OP20) 4대, B5(P3-OP10) 4대. (P2 0대)
        # MODEL_C (5대): B6(P3-OP20) 5대. (P3-OP10 0대)
        eqps = [
            ('B1', 'MODEL_A', '2026051800', 5), ('B3', 'MODEL_A', '2026051800', 5),
            ('B2', 'MODEL_B', '2026051800', 4), ('B5', 'MODEL_B', '2026051800', 4),
            ('B6', 'MODEL_C', '2026051800', 5)
        ]
        for b, m, t, q in eqps:
            self.db.execute(f"INSERT INTO EQP_QTY_INFO VALUES ('{tk}', '{b}', '{m}', '{t}', {q})")
            
        # PLAN (24시간 기준 목표)
        plans = [
            ('P1', 'OP10', '2026051800', '2026051824', 12000), ('P1', 'OP20', '2026051800', '2026051824', 12000),
            ('P2', 'OP10', '2026051800', '2026051824', 8000),  ('P2', 'OP20', '2026051800', '2026051824', 8000),
            ('P3', 'OP10', '2026051800', '2026051824', 5000),  ('P3', 'OP20', '2026051800', '2026051824', 5000)
        ]
        for p, s, st, et, q in plans:
            self.db.execute(f"INSERT INTO PLAN_INFO VALUES ('{tk}', '{p}', '{s}', '{st}', '{et}', {q})")
            
        print("[성공] 벤치마크 데이터셋 DB 적재가 완료되었습니다.")

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

    def generate_expert_data(self, env, num_samples=1000):
        """휴리스틱 룰(UPH 기반 최적화)을 적용한 전문가 데이터 생성"""
        from biz.services.rl.expert import HeuristicExpert
        expert = HeuristicExpert(env)
        
        obs_list = []
        action_list = []
        
        obs, _ = env.reset()
        for _ in range(num_samples):
            action = expert.select_action()
            
            obs_list.append(obs.copy())
            action_list.append(action)
            
            obs, _, done, truncated, _ = env.step(action)
            if done or truncated:
                obs, _ = env.reset()
                
        return np.array(obs_list), np.array(action_list)

    def pretrain_behavior_cloning(self, model, expert_obs, expert_actions, epochs=10, batch_size=32):
        """PyTorch를 이용한 Policy Network 지도 학습 (행동 복제)"""
        print("모방학습(Behavior Cloning) 사전 학습 시작...")
        policy = model.policy
        optimizer = optim.Adam(policy.parameters(), lr=1e-3)
        
        # numpy 배열을 텐서로 변환하고, 모델의 디바이스(CPU/GPU)로 이동
        dataset = TensorDataset(
            torch.tensor(expert_obs, dtype=torch.float32).to(policy.device),
            torch.tensor(expert_actions).to(policy.device)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        policy.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_obs, batch_acts in dataloader:
                # SB3 Policy의 evaluate_actions는 전문가 행동의 log probability를 반환함
                _, log_prob, _ = policy.evaluate_actions(batch_obs, batch_acts)
                
                # NLL Loss (Negative Log Likelihood) = -log_prob의 평균 최소화
                loss = -log_prob.mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        print("모방학습 완료.")

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
    ):
        """단일 데이터 스냅샷으로 24스텝 시뮬레이션 후 지표·전환 횟수 반환."""
        env = SchedulerEnv(data=data, max_steps=max_steps)
        expert = expert_cls(env) if expert_cls is not None else None
        obs, _ = env.reset()
        transfers = 0
        done = False
        while not done:
            if expert is not None:
                action = expert.select_action()
            elif model is not None:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get('transfers'):
                transfers += len(info['transfers'])
            done = terminated or truncated

        env.print_final_summary(method_name=method_name)
        metrics = self._collect_op20_metrics(env)
        metrics['TRANSFERS'] = transfers
        return metrics, env

    def evaluate_on_benchmark_dataset(
        self,
        benchmark_dataset='benchmark_dataset',
        model_path='scheduler_ppo_model',
        max_steps=24,
    ):
        """test/data 벤치마크 데이터셋으로 학습 모델·휴리스틱·정답(Optimal) 성능 비교."""
        from biz.services.rl.test_data_loader import TestDataLoader
        from biz.services.rl.expert import HeuristicExpert, OptimalExpert

        print("\n" + "=" * 80)
        print(f" [벤치마크 데이터셋 성능 비교 — dataset: {benchmark_dataset}]")
        print("=" * 80)

        loader = TestDataLoader()
        data = loader.load_for_env(benchmark_dataset)
        ground_truth = loader.load_ground_truth(benchmark_dataset)

        print("\n[1] 정답지(Optimal Ground Truth) 시뮬레이션...")
        gt_metrics, _ = self._run_simulation_with_policy(
            data, max_steps=max_steps, expert_cls=OptimalExpert, method_name='optimal_test'
        )

        print("\n[2] 휴리스틱(Heuristic Expert) 시뮬레이션...")
        heu_metrics, _ = self._run_simulation_with_policy(
            data, max_steps=max_steps, expert_cls=HeuristicExpert, method_name='heuristic_test'
        )

        print("\n[3] 강화학습(RL PPO) 시뮬레이션...")
        try:
            model = PPO.load(model_path)
        except Exception:
            print(f"[경고] 모델 '{model_path}' 없음 — 무작위 액션으로 평가합니다.")
            model = None
        rl_metrics, env_rl = self._run_simulation_with_policy(
            data, max_steps=max_steps, model=model, method_name='rl_test'
        )

        def _row(label, m):
            return {
                'Method': label,
                'P1 달성률(%)': m.get('P1_OP20_ACHIEVEMENT', 0),
                'P2 달성률(%)': m.get('P2_OP20_ACHIEVEMENT', 0),
                'P3 달성률(%)': m.get('P3_OP20_ACHIEVEMENT', 0),
                '평균 달성률(%)': m.get('AVG_ACHIEVEMENT', 0),
                '장비 전환 횟수': m.get('TRANSFERS', 0),
            }

        comparison_df = pd.DataFrame([
            _row('1. 정답지 (Optimal GT)', gt_metrics),
            _row('2. 휴리스틱 (Expert)', heu_metrics),
            _row('3. 강화학습 (RL PPO)', rl_metrics),
        ])

        print("\n" + "=" * 80)
        print(" [테스트 성능 비교 결과]")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        print("=" * 80)

        ref = ground_truth.setdefault('optimal_reference', {})
        for k, v in gt_metrics.items():
            ref[k] = v

        log_dir = os.path.join(os.getcwd(), 'logs', 'simulation_logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            log_dir, f'benchmark_dataset_eval_{benchmark_dataset}_{timestamp}.xlsx'
        )
        with pd.ExcelWriter(report_path) as writer:
            comparison_df.to_excel(writer, sheet_name='COMPARISON', index=False)
            pd.DataFrame([ref]).to_excel(writer, sheet_name='GROUND_TRUTH', index=False)
        print(f"[성공] 테스트 평가 리포트 저장: {report_path}")

        rl_allocation_df = self._build_final_allocation_df(env_rl)
        rl_achievement_df = self._build_last_process_achievement_df(env_rl, data)
        self.save_inference_summary(
            rule_timekey=ground_truth.get('rule_timekey', 'test'),
            allocation_df=rl_allocation_df,
            achievement_df=rl_achievement_df,
            file_prefix=f'benchmark_inference_{benchmark_dataset}',
        )

        return comparison_df

    def train_model(
        self,
        total_timesteps=10000,
        pretrain_bc=True,
        n_envs=4,
        batch_size=64,
        n_steps=2048,
        rule_timekey=None,
        from_rule_timekey=None,
        to_rule_timekey=None,
        run_test_eval=True,
        benchmark_dataset='benchmark_dataset',
    ):
        """RULE_TIMEKEY 구간(from~to) 또는 단일 키로 학습 후 벤치마크 데이터셋 성능 비교."""
        snapshots = self.fetch_training_snapshots(
            from_rule_timekey=from_rule_timekey,
            to_rule_timekey=to_rule_timekey,
            rule_timekey=rule_timekey,
        )
        bc_data = snapshots[0]

        def make_env():
            if len(snapshots) == 1:
                return Monitor(SchedulerEnv(data=snapshots[0]))
            return Monitor(SnapshotRotationEnv(snapshots))

        if n_envs > 1:
            env = SubprocVecEnv([make_env for _ in range(n_envs)])
        else:
            env = DummyVecEnv([make_env])

        model = PPO("MlpPolicy", env, batch_size=batch_size, n_steps=n_steps, ent_coef=0.01, verbose=1)

        if pretrain_bc:
            single_env = SchedulerEnv(data=bc_data)
            expert_obs, expert_actions = self.generate_expert_data(single_env, num_samples=1000)
            self.pretrain_behavior_cloning(model, expert_obs, expert_actions, epochs=5)

        print("강화학습(PPO) 시작...")
        callback = PlottingCallback(save_path="learning_curve.png")
        model.learn(total_timesteps=total_timesteps, callback=callback)
        model.save("scheduler_ppo_model")
        print("학습 완료 및 모델 저장됨")

        comparison_df = None
        if run_test_eval:
            print("\n[학습 후] 벤치마크 데이터셋 기반 성능 비교를 수행합니다...")
            comparison_df = self.evaluate_on_benchmark_dataset(
                benchmark_dataset=benchmark_dataset
            )
        return comparison_df

    def _build_final_allocation_df(self, env):
        """최종 시뮬레이션 시점의 제품/공정/장비모델별 대수 집계"""
        rows = []
        for p_idx, prod in enumerate(env.products):
            if prod.startswith("PAD_PROD_"):
                continue
            for s_idx, oper in enumerate(env.processes):
                if oper.startswith("PAD_PROC_"):
                    continue
                for m_idx, model in enumerate(env.models):
                    qty = float(env.active_eqp[p_idx, s_idx, m_idx])
                    if qty <= 0:
                        continue
                    rounded_qty = int(round(qty)) if np.isclose(qty, round(qty)) else round(qty, 4)
                    rows.append({
                        "PLAN_PROD_KEY": prod,
                        "OPER_ID": oper,
                        "EQP_MODEL_CD": model,
                        "ALLOCATED_EQP_QTY": rounded_qty
                    })

        allocation_df = pd.DataFrame(
            rows,
            columns=["PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD", "ALLOCATED_EQP_QTY"]
        )
        if not allocation_df.empty:
            allocation_df = allocation_df.sort_values(
                by=["PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD"]
            ).reset_index(drop=True)
        return allocation_df

    def _build_last_process_achievement_df(self, env, data):
        """제품별 마지막 공정의 계획달성률 집계"""
        last_oper_by_prod = {}
        wip_df = data.get('wip_info', pd.DataFrame())

        if not wip_df.empty and {'PLAN_PROD_KEY', 'OPER_ID', 'OPER_SEQ'}.issubset(wip_df.columns):
            tmp = wip_df.copy()
            tmp['OPER_SEQ_NUM'] = pd.to_numeric(tmp['OPER_SEQ'], errors='coerce')
            tmp = tmp.sort_values(by=['PLAN_PROD_KEY', 'OPER_SEQ_NUM', 'OPER_ID'])
            last_rows = tmp.groupby('PLAN_PROD_KEY', as_index=False).tail(1)
            for _, row in last_rows.iterrows():
                last_oper_by_prod[row['PLAN_PROD_KEY']] = row['OPER_ID']

        process_candidates = [proc for proc in env.processes if not proc.startswith("PAD_PROC_")]
        for prod in [p for p in env.products if not p.startswith("PAD_PROD_")]:
            if prod in last_oper_by_prod:
                continue
            if not process_candidates:
                continue
            fallback_oper = max(
                process_candidates,
                key=lambda oper: int(re.search(r"\d+", oper).group()) if re.search(r"\d+", oper) else -1
            )
            last_oper_by_prod[prod] = fallback_oper

        rows = []
        for prod in [p for p in env.products if not p.startswith("PAD_PROD_")]:
            oper = last_oper_by_prod.get(prod)
            if oper is None:
                continue
            p_idx = env.prod_idx.get(prod)
            s_idx = env.proc_idx.get(oper)
            if p_idx is None or s_idx is None:
                continue

            produced_qty = float(env.produced[p_idx, s_idx])
            plan_qty = float(env.plan[p_idx, s_idx])
            rate = (produced_qty / plan_qty * 100.0) if plan_qty > 0 else 0.0

            rows.append({
                "PLAN_PROD_KEY": prod,
                "LAST_OPER_ID": oper,
                "PRODUCED_QTY": round(produced_qty, 4),
                "PLAN_QTY": round(plan_qty, 4),
                "ACHIEVEMENT_RATE(%)": round(rate, 2)
            })

        achievement_df = pd.DataFrame(
            rows,
            columns=["PLAN_PROD_KEY", "LAST_OPER_ID", "PRODUCED_QTY", "PLAN_QTY", "ACHIEVEMENT_RATE(%)"]
        )
        if not achievement_df.empty:
            achievement_df = achievement_df.sort_values(by=["PLAN_PROD_KEY"]).reset_index(drop=True)
        return achievement_df

    def _build_allocation_pivot_df(self, allocation_df):
        """장비모델을 컬럼으로 피벗한 장비 대수 집계"""
        if allocation_df.empty:
            return pd.DataFrame(columns=["PLAN_PROD_KEY", "OPER_ID"])

        pivot_df = allocation_df.pivot_table(
            index=["PLAN_PROD_KEY", "OPER_ID"],
            columns="EQP_MODEL_CD",
            values="ALLOCATED_EQP_QTY",
            aggfunc="sum",
            fill_value=0
        ).reset_index()
        pivot_df.columns.name = None

        model_cols = [col for col in pivot_df.columns if col not in ["PLAN_PROD_KEY", "OPER_ID"]]
        for col in model_cols:
            numeric_values = pd.to_numeric(pivot_df[col], errors='coerce').fillna(0.0)
            if np.all(np.isclose(numeric_values, np.round(numeric_values))):
                pivot_df[col] = np.round(numeric_values).astype(int)
            else:
                pivot_df[col] = numeric_values.round(4)

        pivot_df = pivot_df.sort_values(by=["PLAN_PROD_KEY", "OPER_ID"]).reset_index(drop=True)
        return pivot_df

    def save_inference_summary(self, rule_timekey, allocation_df, achievement_df, file_prefix='inference_summary'):
        """요청된 핵심 결과 요약을 콘솔/엑셀로 저장"""
        allocation_pivot_df = self._build_allocation_pivot_df(allocation_df)

        print("\n[1] 제품 공정별 장비모델별 대수 할당 결과")
        print("-" * 80)
        if allocation_pivot_df.empty:
            print("집계 가능한 장비 할당 결과가 없습니다.")
        else:
            print(allocation_pivot_df.to_string(index=False))
        print("-" * 80)

        print("\n[2] 마지막 공정 기준 제품별 계획달성률")
        print("-" * 80)
        if achievement_df.empty:
            print("집계 가능한 마지막 공정 계획달성률 정보가 없습니다.")
        else:
            print(achievement_df.to_string(index=False))
        print("-" * 80)

        log_dir = os.path.join(os.getcwd(), 'logs', 'simulation_logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_timekey = str(rule_timekey).replace(" ", "_")
        file_path = os.path.join(log_dir, f"{file_prefix}_{safe_timekey}_{timestamp}.xlsx")

        with pd.ExcelWriter(file_path) as writer:
            allocation_pivot_df.to_excel(writer, sheet_name='EQP_ALLOCATION_PIVOT', index=False)
            allocation_df.to_excel(writer, sheet_name='EQP_ALLOCATION', index=False)
            achievement_df.to_excel(writer, sheet_name='LAST_OPER_ACH', index=False)

        print(f"[성공] 추론 요약 리포트가 엑셀로 저장되었습니다: {file_path}")

    def run_inference(self, rule_timekey=None):
        """추론 수행.

        - 조회·출력(RTD_CONV 등) RULE_TIMEKEY: rule_timekey 지정, 미지정 시 WIP_INFO MAX(RULE_TIMEKEY)
        """
        resolved_timekey = self._resolve_rule_timekey(rule_timekey)
        print(f"[추론] RULE_TIMEKEY={resolved_timekey} (입력 스냅샷·결과 출력 공통)")
        data = self.fetch_data(rule_timekey=resolved_timekey)
        env = SchedulerEnv(data=data)

        try:
            model = PPO.load("scheduler_ppo_model")
        except Exception:
            print("저장된 모델이 없습니다. 임의의 액션으로 시뮬레이션 합니다.")
            model = None

        obs, _ = env.reset()
        done = False

        results = []

        while not done:
            if model:
                action, _states = model.predict(obs, deterministic=True)
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
        if not logs:
            return
        df = pd.DataFrame(logs)
        print(f"\n[시뮬레이션 생산 로그] 총 {len(df)}건")
        print("-" * 60)
        print(df.to_string(index=False))
        print("-" * 60)
        
        # Save to Excel
        log_dir = os.path.join(os.getcwd(), 'logs', 'simulation_logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(log_dir, f'production_log_{timestamp}.xlsx')
        df.to_excel(file_path, index=False)
        print(f"[성공] 생산 로그가 엑셀로 저장되었습니다: {file_path}")

    def save_results(self, results):
        if not results:
            print("\n[장비 전환 액션] 도출된 전환 액션이 없습니다.")
            return
            
        df = pd.DataFrame(results)
        print(f"\n[장비 전환 액션] 총 {len(df)}건 도출")
        print("-" * 80)
        print(df.to_string(index=False))
        print("-" * 80)
        
        # Save to Excel
        log_dir = os.path.join(os.getcwd(), 'logs', 'simulation_logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(log_dir, f'action_log_{timestamp}.xlsx')
        df.to_excel(file_path, index=False)
        print(f"[성공] 액션 로그가 엑셀로 저장되었습니다: {file_path}")

    def run_benchmark_evaluation(self, total_timesteps=10000):
        """벤치마크 데이터셋 기준 Optimal vs 휴리스틱 vs RL 성능 비교 (학습 포함)."""
        print("\n" + "="*80)
        print(" [벤치마크 데이터셋 평가 리포트 (Optimal vs Heuristic vs RL)]")
        print("="*80)
        
        # 1. 벤치마크 데이터셋 DB 적재 및 로드
        self.init_benchmark_dataset_scenario()
        data = self.fetch_data()
        
        # 2. 정답지 (Ground Truth / Optimal) 시뮬레이션
        print("\n[1단계] 정답지(Optimal Ground Truth) 시뮬레이션 진행 중...")
        from biz.services.rl.expert import OptimalExpert
        env_gt = SchedulerEnv(data=data, max_steps=24)
        gt_expert = OptimalExpert(env_gt)
        
        obs, _ = env_gt.reset()
        gt_transfers = 0
        done = False
        while not done:
            action = gt_expert.select_action()
            obs, reward, terminated, truncated, info = env_gt.step(action)
            if 'transfers' in info and info['transfers']:
                gt_transfers += len(info['transfers'])
            done = terminated or truncated
            
        print("\n[정답지 시뮬레이션 최종결과 상세]")
        env_gt.print_final_summary(method_name="optimal")
            
        # 정답지 결과 집계 (최종 공정 OP20 기준)
        gt_results = {}
        for p_idx, p_name in enumerate(env_gt.products):
            if p_name in ['P1', 'P2', 'P3']:
                s_idx = env_gt.proc_idx.get('OP20')
                if s_idx is not None:
                    prod_qty = env_gt.produced[p_idx, s_idx]
                    plan_qty = env_gt.plan[p_idx, s_idx]
                    rate = (prod_qty / plan_qty * 100.0) if plan_qty > 0 else 0.0
                    gt_results[f"{p_name}_OP20_ACHIEVEMENT"] = round(rate, 2)
                    
        gt_avg = round(sum(gt_results.values()) / len(gt_results), 2) if gt_results else 0.0
        gt_results['AVG_ACHIEVEMENT'] = gt_avg
        gt_results['TRANSFERS'] = gt_transfers
        
        print(f" - 정답지 평균 계획달성률: {gt_avg}% / 총 장비 전환 횟수: {gt_transfers}회")
        
        # 3. 일반 휴리스틱 (Heuristic Expert) 시뮬레이션
        print("\n[2단계] 일반 휴리스틱(Heuristic Expert) 시뮬레이션 진행 중 (24 Steps)...")
        from biz.services.rl.expert import HeuristicExpert
        env_heu = SchedulerEnv(data=data, max_steps=24)
        expert = HeuristicExpert(env_heu)
        
        obs, _ = env_heu.reset()
        heu_transfers = 0
        done = False
        while not done:
            action = expert.select_action()
            obs, reward, terminated, truncated, info = env_heu.step(action)
            if 'transfers' in info and info['transfers']:
                heu_transfers += len(info['transfers'])
            done = terminated or truncated
            
        print("\n[휴리스틱 시뮬레이션 최종결과 상세]")
        env_heu.print_final_summary(method_name="heuristic")
            
        # 휴리스틱 결과 집계 (최종 공정 OP20 기준)
        heu_results = {}
        for p_idx, p_name in enumerate(env_heu.products):
            if p_name in ['P1', 'P2', 'P3']:
                s_idx = env_heu.proc_idx.get('OP20')
                if s_idx is not None:
                    prod_qty = env_heu.produced[p_idx, s_idx]
                    plan_qty = env_heu.plan[p_idx, s_idx]
                    rate = (prod_qty / plan_qty * 100.0) if plan_qty > 0 else 0.0
                    heu_results[f"{p_name}_OP20_ACHIEVEMENT"] = round(rate, 2)
                    
        heu_avg = round(sum(heu_results.values()) / len(heu_results), 2) if heu_results else 0.0
        heu_results['AVG_ACHIEVEMENT'] = heu_avg
        heu_results['TRANSFERS'] = heu_transfers
        
        print(f" - 휴리스틱 평균 계획달성률: {heu_avg}% / 총 장비 전환 횟수: {heu_transfers}회")
        
        # 4. 강화학습 (RL PPO) 시뮬레이션
        print("\n[3단계] 강화학습(RL PPO) 모델 학습 및 시뮬레이션 진행 중...")
        # 학습 수행
        self.train_model(total_timesteps=total_timesteps, pretrain_bc=True, n_envs=1)
        
        env_rl = SchedulerEnv(data=data, max_steps=24)
        try:
            model = PPO.load("scheduler_ppo_model")
        except Exception:
            model = None
            
        obs, _ = env_rl.reset()
        rl_transfers = 0
        done = False
        while not done:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env_rl.action_space.sample()
            obs, reward, terminated, truncated, info = env_rl.step(int(action))
            if 'transfers' in info and info['transfers']:
                rl_transfers += len(info['transfers'])
            done = terminated or truncated
            
        print("\n[강화학습 시뮬레이션 최종결과 상세]")
        env_rl.print_final_summary(method_name="rl")
            
        # RL 결과 집계 (최종 공정 OP20 기준)
        rl_results = {}
        for p_idx, p_name in enumerate(env_rl.products):
            if p_name in ['P1', 'P2', 'P3']:
                s_idx = env_rl.proc_idx.get('OP20')
                if s_idx is not None:
                    prod_qty = env_rl.produced[p_idx, s_idx]
                    plan_qty = env_rl.plan[p_idx, s_idx]
                    rate = (prod_qty / plan_qty * 100.0) if plan_qty > 0 else 0.0
                    rl_results[f"{p_name}_OP20_ACHIEVEMENT"] = round(rate, 2)
                    
        rl_avg = round(sum(rl_results.values()) / len(rl_results), 2) if rl_results else 0.0
        rl_results['AVG_ACHIEVEMENT'] = rl_avg
        rl_results['TRANSFERS'] = rl_transfers
        
        print(f" - 강화학습 평균 계획달성률: {rl_avg}% / 총 장비 전환 횟수: {rl_transfers}회")
        
        # 5. 최종 비교 테이블 생성 및 출력
        print("\n" + "="*80)
        print(" [최종 벤치마크 비교 결과 (최종 공정 OP20 기준 계획달성률)]")
        print("="*80)
        
        comparison_df = pd.DataFrame([
            {
                'Method': '1. 정답지 (Optimal GT)', 
                'P1 달성률(%)': gt_results.get('P1_OP20_ACHIEVEMENT', 0),
                'P2 달성률(%)': gt_results.get('P2_OP20_ACHIEVEMENT', 0),
                'P3 달성률(%)': gt_results.get('P3_OP20_ACHIEVEMENT', 0),
                '평균 달성률(%)': gt_results.get('AVG_ACHIEVEMENT', 0),
                '장비 전환 횟수': gt_results.get('TRANSFERS', 0)
            },
            {
                'Method': '2. 일반 휴리스틱 (Expert)', 
                'P1 달성률(%)': heu_results.get('P1_OP20_ACHIEVEMENT', 0),
                'P2 달성률(%)': heu_results.get('P2_OP20_ACHIEVEMENT', 0),
                'P3 달성률(%)': heu_results.get('P3_OP20_ACHIEVEMENT', 0),
                '평균 달성률(%)': heu_results.get('AVG_ACHIEVEMENT', 0),
                '장비 전환 횟수': heu_results.get('TRANSFERS', 0)
            },
            {
                'Method': '3. 강화학습 (RL PPO)', 
                'P1 달성률(%)': rl_results.get('P1_OP20_ACHIEVEMENT', 0),
                'P2 달성률(%)': rl_results.get('P2_OP20_ACHIEVEMENT', 0),
                'P3 달성률(%)': rl_results.get('P3_OP20_ACHIEVEMENT', 0),
                '평균 달성률(%)': rl_results.get('AVG_ACHIEVEMENT', 0),
                '장비 전환 횟수': rl_results.get('TRANSFERS', 0)
            }
        ])
        
        print(comparison_df.to_string(index=False))
        print("="*80)

        rl_allocation_df = self._build_final_allocation_df(env_rl)
        rl_last_oper_achievement_df = self._build_last_process_achievement_df(env_rl, data)
        self.save_inference_summary(
            rule_timekey='benchmark',
            allocation_df=rl_allocation_df,
            achievement_df=rl_last_oper_achievement_df,
            file_prefix='benchmark_dataset_inference_summary'
        )
        
        # 엑셀 리포트 저장
        log_dir = os.path.join(os.getcwd(), 'logs', 'simulation_logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(log_dir, f'benchmark_dataset_report_{timestamp}.xlsx')
        comparison_df.to_excel(file_path, index=False)
        print(f"[성공] 벤치마크 데이터셋 평가 리포트가 저장되었습니다: {file_path}")
        
        return comparison_df

