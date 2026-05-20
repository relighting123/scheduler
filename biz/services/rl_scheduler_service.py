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
class RLSchedulerService:
    def __init__(self, db_manager):
        # db_manager는 Core에서 주입받는다고 가정
        self.db = db_manager

    def init_db_scenario(self):
        """휴리스틱 함정(Trap) 시나리오용 DB 초기화 (DROP -> CREATE -> INSERT)"""
        print("\n[DB 설정] DB 시나리오 초기화를 시작합니다 (Heuristic Trap Scenario)...")
        tables = [
            "WIP_INFO", "UPH_INFO", "EQP_QTY_INFO", "AVAIL_INFO", 
            "BATCH_TOOL_INFO", "TOOL_QTY_INFO", "PLAN_INFO", "RTD_CONV_INF"
        ]
        
        # 1. DROP Tables
        for table in tables:
            try:
                self.db.execute(f"DROP TABLE {table}")
            except Exception:
                pass # 테이블이 없으면 무시
                
        # 2. CREATE Tables
        self.db.execute("""
            CREATE TABLE WIP_INFO (
                PLAN_PROD_KEY VARCHAR2(50), OPER_ID VARCHAR2(50), OPER_SEQ NUMBER, WIP_QTY NUMBER
            )
        """)
        self.db.execute("""
            CREATE TABLE UPH_INFO (
                PLAN_PROD_KEY VARCHAR2(50), OPER_ID VARCHAR2(50), EQP_MODEL_CD VARCHAR2(50), UPH NUMBER
            )
        """)
        self.db.execute("""
            CREATE TABLE EQP_QTY_INFO (
                BATCH_ID VARCHAR2(50), EQP_MODEL_CD VARCHAR2(50), TIME_SLOT VARCHAR2(50), EQP_QTY NUMBER
            )
        """)
        self.db.execute("""
            CREATE TABLE AVAIL_INFO (
                PLAN_PROD_KEY VARCHAR2(50), OPER_ID VARCHAR2(50), EQP_MODEL_CD VARCHAR2(50), AVAIL_YN VARCHAR2(10)
            )
        """)
        self.db.execute("""
            CREATE TABLE BATCH_TOOL_INFO (
                BATCH_ID VARCHAR2(50), PLAN_PROD_KEY VARCHAR2(50), OPER_ID VARCHAR2(50)
            )
        """)
        self.db.execute("""
            CREATE TABLE TOOL_QTY_INFO (
                BATCH_ID VARCHAR2(50), EQP_MODEL_CD VARCHAR2(50), TOOL_QTY NUMBER
            )
        """)
        self.db.execute("""
            CREATE TABLE PLAN_INFO (
                PLAN_PROD_KEY VARCHAR2(50), OPER_ID VARCHAR2(50), START_TIME VARCHAR2(50), END_TIME VARCHAR2(50), PLAN_QTY NUMBER
            )
        """)
        
        # 3. INSERT Data (Trap Scenario)
        # WIP: OP10 = 5000 (Trap trigger), OP20 = 500
        self.db.execute("INSERT INTO WIP_INFO VALUES ('P1', 'OP10', 10, 5000)")
        self.db.execute("INSERT INTO WIP_INFO VALUES ('P1', 'OP20', 20, 500)")
        
        # UPH: 100/hr for both operations
        self.db.execute("INSERT INTO UPH_INFO VALUES ('P1', 'OP10', 'MODEL_A', 100)")
        self.db.execute("INSERT INTO UPH_INFO VALUES ('P1', 'OP20', 'MODEL_A', 100)")
        
        # EQP_QTY (Initial allocation): 5 units perfectly balanced
        self.db.execute("INSERT INTO EQP_QTY_INFO VALUES ('B1', 'MODEL_A', '2026051800', 5)")
        self.db.execute("INSERT INTO EQP_QTY_INFO VALUES ('B2', 'MODEL_A', '2026051800', 5)")
        
        # AVAIL
        self.db.execute("INSERT INTO AVAIL_INFO VALUES ('P1', 'OP10', 'MODEL_A', 'Y')")
        self.db.execute("INSERT INTO AVAIL_INFO VALUES ('P1', 'OP20', 'MODEL_A', 'Y')")
        
        # BATCH
        self.db.execute("INSERT INTO BATCH_TOOL_INFO VALUES ('B1', 'P1', 'OP10')")
        self.db.execute("INSERT INTO BATCH_TOOL_INFO VALUES ('B2', 'P1', 'OP20')")
        
        # PLAN
        self.db.execute("INSERT INTO PLAN_INFO VALUES ('P1', 'OP10', '2026051800', '2026051824', 4000)")
        self.db.execute("INSERT INTO PLAN_INFO VALUES ('P1', 'OP20', '2026051800', '2026051824', 4000)")
        
        print("[성공] DB 시나리오 초기화가 완료되었습니다 (WIP_INFO 등 7개 테이블).")

    def init_combinatorial_scenario(self):
        """다품종/다모델/전용모델/UPH 분산 등 조합최적화 문제 시나리오용 DB 초기화"""
        print("\n[DB 설정] 조합최적화 벤치마크 시나리오 초기화를 시작합니다 (Combinatorial Optimization Scenario)...")
        tables = [
            "WIP_INFO", "UPH_INFO", "EQP_QTY_INFO", "AVAIL_INFO", 
            "BATCH_TOOL_INFO", "TOOL_QTY_INFO", "PLAN_INFO", "RTD_CONV_INF"
        ]
        
        # 1. DROP Tables
        for table in tables:
            try:
                self.db.execute(f"DROP TABLE {table}")
            except Exception:
                pass # 테이블이 없으면 무시
                
        # 2. CREATE Tables
        self.db.execute("""
            CREATE TABLE WIP_INFO (
                PLAN_PROD_KEY VARCHAR2(50), OPER_ID VARCHAR2(50), OPER_SEQ NUMBER, WIP_QTY NUMBER
            )
        """)
        self.db.execute("""
            CREATE TABLE UPH_INFO (
                PLAN_PROD_KEY VARCHAR2(50), OPER_ID VARCHAR2(50), EQP_MODEL_CD VARCHAR2(50), UPH NUMBER
            )
        """)
        self.db.execute("""
            CREATE TABLE EQP_QTY_INFO (
                BATCH_ID VARCHAR2(50), EQP_MODEL_CD VARCHAR2(50), TIME_SLOT VARCHAR2(50), EQP_QTY NUMBER
            )
        """)
        self.db.execute("""
            CREATE TABLE AVAIL_INFO (
                PLAN_PROD_KEY VARCHAR2(50), OPER_ID VARCHAR2(50), EQP_MODEL_CD VARCHAR2(50), AVAIL_YN VARCHAR2(10)
            )
        """)
        self.db.execute("""
            CREATE TABLE BATCH_TOOL_INFO (
                BATCH_ID VARCHAR2(50), PLAN_PROD_KEY VARCHAR2(50), OPER_ID VARCHAR2(50)
            )
        """)
        self.db.execute("""
            CREATE TABLE TOOL_QTY_INFO (
                BATCH_ID VARCHAR2(50), EQP_MODEL_CD VARCHAR2(50), TOOL_QTY NUMBER
            )
        """)
        self.db.execute("""
            CREATE TABLE PLAN_INFO (
                PLAN_PROD_KEY VARCHAR2(50), OPER_ID VARCHAR2(50), START_TIME VARCHAR2(50), END_TIME VARCHAR2(50), PLAN_QTY NUMBER
            )
        """)
        
        # 3. INSERT Data (Combinatorial Scenario)
        # WIP: 충분한 재공 부여
        wips = [
            ('P1', 'OP10', 10, 15000), ('P1', 'OP20', 20, 2000),
            ('P2', 'OP10', 10, 10000), ('P2', 'OP20', 20, 1000),
            ('P3', 'OP10', 10, 6000),  ('P3', 'OP20', 20, 500)
        ]
        for p, s, seq, q in wips:
            self.db.execute(f"INSERT INTO WIP_INFO VALUES ('{p}', '{s}', {seq}, {q})")
            
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
            self.db.execute(f"INSERT INTO UPH_INFO VALUES ('{p}', '{s}', '{m}', {u})")
            
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
            self.db.execute(f"INSERT INTO AVAIL_INFO VALUES ('{p}', '{s}', '{m}', '{a}')")
            
        # BATCH
        batches = [
            ('B1', 'P1', 'OP10'), ('B2', 'P1', 'OP20'),
            ('B3', 'P2', 'OP10'), ('B4', 'P2', 'OP20'),
            ('B5', 'P3', 'OP10'), ('B6', 'P3', 'OP20')
        ]
        for b, p, s in batches:
            self.db.execute(f"INSERT INTO BATCH_TOOL_INFO VALUES ('{b}', '{p}', '{s}')")
            
        # TOOL QTY
        for b in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']:
            for m in ['MODEL_A', 'MODEL_B', 'MODEL_C']:
                self.db.execute(f"INSERT INTO TOOL_QTY_INFO VALUES ('{b}', '{m}', 20)")
                
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
            self.db.execute(f"INSERT INTO EQP_QTY_INFO VALUES ('{b}', '{m}', '{t}', {q})")
            
        # PLAN (24시간 기준 목표)
        plans = [
            ('P1', 'OP10', '2026051800', '2026051824', 12000), ('P1', 'OP20', '2026051800', '2026051824', 12000),
            ('P2', 'OP10', '2026051800', '2026051824', 8000),  ('P2', 'OP20', '2026051800', '2026051824', 8000),
            ('P3', 'OP10', '2026051800', '2026051824', 5000),  ('P3', 'OP20', '2026051800', '2026051824', 5000)
        ]
        for p, s, st, et, q in plans:
            self.db.execute(f"INSERT INTO PLAN_INFO VALUES ('{p}', '{s}', '{st}', '{et}', {q})")
            
        print("[성공] 조합최적화 벤치마크용 DB 시나리오 초기화가 완료되었습니다.")

    def fetch_data(self):
        """DB에서 데이터를 조회해 옵니다."""
        try:
            wip_data = pd.DataFrame(self.db.select_list("SELECT PLAN_PROD_KEY, OPER_ID, OPER_SEQ, WIP_QTY FROM WIP_INFO"), columns=['PLAN_PROD_KEY', 'OPER_ID', 'OPER_SEQ', 'WIP_QTY'])
            uph_data = pd.DataFrame(self.db.select_list("SELECT PLAN_PROD_KEY, OPER_ID, EQP_MODEL_CD, UPH FROM UPH_INFO"), columns=['PLAN_PROD_KEY', 'OPER_ID', 'EQP_MODEL_CD', 'UPH'])
            eqp_qty_data = pd.DataFrame(self.db.select_list("SELECT BATCH_ID, EQP_MODEL_CD, TIME_SLOT, EQP_QTY FROM EQP_QTY_INFO"), columns=['BATCH_ID', 'EQP_MODEL_CD', 'TIME_SLOT', 'EQP_QTY'])
            avail_data = pd.DataFrame(self.db.select_list("SELECT PLAN_PROD_KEY, OPER_ID, EQP_MODEL_CD, AVAIL_YN FROM AVAIL_INFO"), columns=['PLAN_PROD_KEY', 'OPER_ID', 'EQP_MODEL_CD', 'AVAIL_YN'])
            batch_tool_data = pd.DataFrame(self.db.select_list("SELECT BATCH_ID, PLAN_PROD_KEY, OPER_ID FROM BATCH_TOOL_INFO"), columns=['BATCH_ID', 'PLAN_PROD_KEY', 'OPER_ID'])
            tool_qty_data = pd.DataFrame(self.db.select_list("SELECT BATCH_ID, EQP_MODEL_CD, TOOL_QTY FROM TOOL_QTY_INFO"), columns=['BATCH_ID', 'EQP_MODEL_CD', 'TOOL_QTY'])
            plan_data = pd.DataFrame(self.db.select_list("SELECT PLAN_PROD_KEY, OPER_ID, START_TIME, END_TIME, PLAN_QTY FROM PLAN_INFO"), columns=['PLAN_PROD_KEY', 'OPER_ID', 'START_TIME', 'END_TIME', 'PLAN_QTY'])
            
            # DB가 비어있을 경우 에러 방지를 위해 기본 컬럼 지정
            if wip_data.empty: wip_data = pd.DataFrame(columns=['PLAN_PROD_KEY', 'OPER_ID', 'OPER_SEQ', 'WIP_QTY'])
            if uph_data.empty: uph_data = pd.DataFrame(columns=['PLAN_PROD_KEY', 'OPER_ID', 'EQP_MODEL_CD', 'UPH'])
            if eqp_qty_data.empty: eqp_qty_data = pd.DataFrame(columns=['BATCH_ID', 'EQP_MODEL_CD', 'TIME_SLOT', 'EQP_QTY'])
            if avail_data.empty: avail_data = pd.DataFrame(columns=['PLAN_PROD_KEY', 'OPER_ID', 'EQP_MODEL_CD', 'AVAIL_YN'])
            if batch_tool_data.empty: batch_tool_data = pd.DataFrame(columns=['BATCH_ID', 'PLAN_PROD_KEY', 'OPER_ID'])
            if tool_qty_data.empty: tool_qty_data = pd.DataFrame(columns=['BATCH_ID', 'EQP_MODEL_CD', 'TOOL_QTY'])
            if plan_data.empty: plan_data = pd.DataFrame(columns=['PLAN_PROD_KEY', 'OPER_ID', 'START_TIME', 'END_TIME', 'PLAN_QTY'])
            
            print("[성공] DB에서 성공적으로 스케줄링 기초 데이터를 조회했습니다.")
            
        except Exception as e:
            print(f"[경고] DB 연동 실패 (또는 테이블 없음): {e}")
            print("데이터를 조회할 수 없습니다. DB 초기화(init_db_scenario)가 올바르게 수행되었는지 확인하세요.")
            # 실패 시 빈 DataFrame 반환
            wip_data = pd.DataFrame(columns=['PLAN_PROD_KEY', 'OPER_ID', 'OPER_SEQ', 'WIP_QTY'])
            uph_data = pd.DataFrame(columns=['PLAN_PROD_KEY', 'OPER_ID', 'EQP_MODEL_CD', 'UPH'])
            eqp_qty_data = pd.DataFrame(columns=['BATCH_ID', 'EQP_MODEL_CD', 'TIME_SLOT', 'EQP_QTY'])
            avail_data = pd.DataFrame(columns=['PLAN_PROD_KEY', 'OPER_ID', 'EQP_MODEL_CD', 'AVAIL_YN'])
            batch_tool_data = pd.DataFrame(columns=['BATCH_ID', 'PLAN_PROD_KEY', 'OPER_ID'])
            tool_qty_data = pd.DataFrame(columns=['BATCH_ID', 'EQP_MODEL_CD', 'TOOL_QTY'])
            plan_data = pd.DataFrame(columns=['PLAN_PROD_KEY', 'OPER_ID', 'START_TIME', 'END_TIME', 'PLAN_QTY'])

        return {
            'wip_info': wip_data,
            'uph_info': uph_data,
            'eqp_qty_info': eqp_qty_data,
            'avail_info': avail_data,
            'batch_tool_info': batch_tool_data,
            'tool_qty_info': tool_qty_data,
            'plan_info': plan_data
        }

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

    def train_model(self, total_timesteps=10000, pretrain_bc=True, n_envs=4, batch_size=64, n_steps=2048):
        data = self.fetch_data()
        
        def make_env():
            return Monitor(SchedulerEnv(data=data))
            
        if n_envs > 1:
            env = SubprocVecEnv([make_env for _ in range(n_envs)])
        else:
            env = DummyVecEnv([make_env])
        
        model = PPO("MlpPolicy", env, batch_size=batch_size, n_steps=n_steps, ent_coef=0.01, verbose=1)
        
        if pretrain_bc:
            single_env = SchedulerEnv(data=data)
            expert_obs, expert_actions = self.generate_expert_data(single_env, num_samples=1000)
            self.pretrain_behavior_cloning(model, expert_obs, expert_actions, epochs=5)
            
        print("강화학습(PPO) 시작...")
        callback = PlottingCallback(save_path="learning_curve.png")
        model.learn(total_timesteps=total_timesteps, callback=callback)
        model.save("scheduler_ppo_model")
        print("학습 완료 및 모델 저장됨")

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
        """추론을 수행하고 결과를 DB에 저장합니다."""
        data = self.fetch_data()
        env = SchedulerEnv(data=data)
        
        try:
            model = PPO.load("scheduler_ppo_model")
        except:
            print("저장된 모델이 없습니다. 임의의 액션으로 시뮬레이션 합니다.")
            model = None
            
        obs, _ = env.reset()
        done = False
        
        results = []
        if not rule_timekey or rule_timekey == 'N/A':
            rule_timekey = datetime.now().strftime("%Y%m%d%H%M%S%f")[:16] # 16자리 맞춤
        
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
                        'RULE_TIMEKEY': rule_timekey,
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
        self.save_inference_summary(rule_timekey, allocation_df, achievement_df)
            
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

    def run_combinatorial_benchmark(self, total_timesteps=10000):
        """정답지(Optimal) vs 일반 휴리스틱 vs RL 모델의 성능 비교 벤치마크 수행"""
        print("\n" + "="*80)
        print(" [조합최적화 스케줄링 벤치마크 리포트 생성 (Optimal vs Heuristic vs RL)]")
        print("="*80)
        
        # 1. 시나리오 초기화 및 데이터 로드
        self.init_combinatorial_scenario()
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
            file_prefix='combinatorial_inference_summary'
        )
        
        # 엑셀 리포트 저장
        log_dir = os.path.join(os.getcwd(), 'logs', 'simulation_logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(log_dir, f'combinatorial_benchmark_{timestamp}.xlsx')
        comparison_df.to_excel(file_path, index=False)
        print(f"[성공] 조합최적화 벤치마크 리포트가 엑셀로 저장되었습니다: {file_path}")
        
        return comparison_df
