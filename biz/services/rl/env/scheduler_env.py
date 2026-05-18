import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict

class SchedulerEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data: Dict[str, pd.DataFrame], max_prods: int = 10, max_procs: int = 10, max_steps: int = 24):
        super(SchedulerEnv, self).__init__()
        self.data = data
        self.max_prods = max_prods
        self.max_procs = max_procs
        self.max_steps = max_steps
        
        self.products = []
        self.processes = []
        self.models = []
        
        self._discover_entities()
        
        self.num_prods = len(self.products)
        self.num_procs = len(self.processes)
        self.num_models = len(self.models)
        
        # Ensure we have at least 1 dimension
        if self.num_prods == 0: self.num_prods = 1
        if self.num_procs == 0: self.num_procs = 1
        if self.num_models == 0: self.num_models = 1

        self.action_space = spaces.Discrete(self.num_prods * self.num_procs * self.num_models + 1)
        
        self.obs_dim = (self.num_prods * self.num_procs * 8) + 2
        self.observation_space = spaces.Box(low=0, high=1000, shape=(self.obs_dim,), dtype=np.float32)

        self.reset()

    def _discover_entities(self):
        wip_df = self.data.get('wip_info', pd.DataFrame())
        uph_df = self.data.get('uph_info', pd.DataFrame())
        
        prods = set()
        procs = set()
        models = set()
        
        if not wip_df.empty:
            prods.update(wip_df['PLAN_PROD_KEY'].unique())
            procs.update(wip_df['OPER_ID'].unique())
            
        if not uph_df.empty:
            prods.update(uph_df['PLAN_PROD_KEY'].unique())
            procs.update(uph_df['OPER_ID'].unique())
            models.update(uph_df['EQP_MODEL_CD'].unique())
            
        self.products = sorted(list(prods))[:self.max_prods]
        self.processes = sorted(list(procs))[:self.max_procs]
        self.models = sorted(list(models))
        
        while len(self.products) < self.max_prods:
            self.products.append(f"PAD_PROD_{len(self.products)}")
        while len(self.processes) < self.max_procs:
            self.processes.append(f"PAD_PROC_{len(self.processes)}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        self.wip = np.zeros((self.num_prods, self.num_procs))
        self.plan = np.zeros((self.num_prods, self.num_procs))
        self.produced = np.zeros((self.num_prods, self.num_procs))

        self.active_eqp = np.zeros((self.num_prods, self.num_procs, self.num_models))
        self.target_eqp = np.zeros((self.num_prods, self.num_procs, self.num_models))
        self.st_matrix = np.zeros((self.num_prods, self.num_procs, self.num_models))
        
        self.prod_idx = {p: i for i, p in enumerate(self.products)}
        self.proc_idx = {p: i for i, p in enumerate(self.processes)}
        self.model_idx = {m: i for i, m in enumerate(self.models)}
        
        # Load from df
        wip_df = self.data.get('wip_info', pd.DataFrame())
        if not wip_df.empty:
            for _, row in wip_df.iterrows():
                p, s = row['PLAN_PROD_KEY'], row['OPER_ID']
                if p in self.prod_idx and s in self.proc_idx:
                    self.wip[self.prod_idx[p], self.proc_idx[s]] += float(row['WIP_QTY'])
                    
        plan_df = self.data.get('plan_info', pd.DataFrame())
        if not plan_df.empty:
            for _, row in plan_df.iterrows():
                p, s = row['PLAN_PROD_KEY'], row['OPER_ID']
                if p in self.prod_idx and s in self.proc_idx:
                    self.plan[self.prod_idx[p], self.proc_idx[s]] += float(row['PLAN_QTY'])
                    
        uph_df = self.data.get('uph_info', pd.DataFrame())
        if not uph_df.empty:
            for _, row in uph_df.iterrows():
                p, s, m = row['PLAN_PROD_KEY'], row['OPER_ID'], row['EQP_MODEL_CD']
                if p in self.prod_idx and s in self.proc_idx and m in self.model_idx:
                    uph = float(row['UPH'])
                    if uph > 0:
                        self.st_matrix[self.prod_idx[p], self.proc_idx[s], self.model_idx[m]] = 60.0 / uph
                        
        # Initialize active equipment from DB
        batch_df = self.data.get('batch_tool_info', pd.DataFrame())
        eqp_qty_df = self.data.get('eqp_qty_info', pd.DataFrame())
        
        has_eqp_data = False
        if not batch_df.empty and not eqp_qty_df.empty:
            merged = pd.merge(batch_df, eqp_qty_df, on='BATCH_ID')
            if not merged.empty:
                has_eqp_data = True
                for _, row in merged.iterrows():
                    p, s, m = row['PLAN_PROD_KEY'], row['OPER_ID'], row['EQP_MODEL_CD']
                    if p in self.prod_idx and s in self.proc_idx and m in self.model_idx:
                        self.active_eqp[self.prod_idx[p], self.proc_idx[s], self.model_idx[m]] += float(row['EQP_QTY'])
                        
        if not has_eqp_data:
            # Dummy init for active equipment if there's no data
            for p in range(self.num_prods):
                for s in range(self.num_procs):
                    for m in range(self.num_models):
                        if self.st_matrix[p, s, m] > 0:
                            self.active_eqp[p, s, m] = 1.0
                            
        # IDLE equipment is 0 initially (all eqp is assigned in the DB)
        self.idle_eqp = np.zeros(self.num_models)
        self.production_logs = []
        return self._get_obs(), {}

    def _get_obs(self):
        # Simplified normalization for observation
        wip_norm = (self.wip / 1000.0).flatten()
        active_norm = (self.active_eqp.sum(axis=2) / 100.0).flatten()
        target_norm = (self.target_eqp.sum(axis=2) / 100.0).flatten()
        co_norm = np.zeros((self.num_prods, self.num_procs)).flatten() # simplified
        produced_ratio = (self.produced / (self.plan + 1e-6)).flatten()
        
        st_per_pp = np.zeros((self.num_prods, self.num_procs))
        for i in range(self.num_prods):
            for j in range(self.num_procs):
                sts = self.st_matrix[i, j, :]
                positive_sts = sts[sts > 0]
                st_per_pp[i, j] = positive_sts.min() if positive_sts.size > 0 else 0.0
                
        st_norm = (st_per_pp / 60.0).flatten()
        plan_norm = (self.plan / 1000.0).flatten()
        wip_plan_ratio = (self.wip / (self.plan + 1e-6)).flatten()
        
        obs = np.concatenate([
            wip_norm, active_norm, target_norm, co_norm,
            produced_ratio, st_norm, plan_norm, wip_plan_ratio,
            [1.0], [float(self.current_step) / self.max_steps]
        ])
        return obs.astype(np.float32)

    def step(self, action):
        transfers = []
        
        if action > 0:
            target_idx = action - 1
            t_prod = target_idx // (self.num_procs * self.num_models)
            rem = target_idx % (self.num_procs * self.num_models)
            t_proc = rem // self.num_models
            t_model = rem % self.num_models
            
            # Check if this assignment is feasible
            if self.st_matrix[t_prod, t_proc, t_model] > 0:
                moved = False
                # Try pulling from IDLE first
                if self.idle_eqp[t_model] > 0:
                    self.idle_eqp[t_model] -= 1
                    self.target_eqp[t_prod, t_proc, t_model] += 1
                    moved = True
                    transfers.append({
                        'FROM_PROD': 'IDLE', 'FROM_PROC': 'IDLE',
                        'TO_PROD': self.products[t_prod], 'TO_PROC': self.processes[t_proc],
                        'MODEL': self.models[t_model]
                    })
                    
                # If no IDLE, pull from another active node
                if not moved:
                    for p in range(self.num_prods):
                        for s in range(self.num_procs):
                            if self.active_eqp[p, s, t_model] > 0 and (p != t_prod or s != t_proc):
                                self.active_eqp[p, s, t_model] -= 1
                                self.target_eqp[t_prod, t_proc, t_model] += 1
                                moved = True
                                transfers.append({
                                    'FROM_PROD': self.products[p], 'FROM_PROC': self.processes[s],
                                    'TO_PROD': self.products[t_prod], 'TO_PROC': self.processes[t_proc],
                                    'MODEL': self.models[t_model]
                                })
                                break
                        if moved: break
                    
        # Simulate production
        step_production = 0.0
        for p in range(self.num_prods):
            for s in range(self.num_procs):
                capacity = 0.0
                active_count = np.sum(self.active_eqp[p, s, :])
                transition_count = np.sum(self.target_eqp[p, s, :])
                
                for m in range(self.num_models):
                    st_val = self.st_matrix[p, s, m]
                    if st_val > 0.0:
                        capacity += (60.0 / st_val) * self.active_eqp[p, s, m]
                
                actual_produce = min(capacity, self.wip[p, s])
                
                # 가동률 (Utilization Rate)
                utilization = (actual_produce / capacity * 100.0) if capacity > 0 else 0.0
                
                self.produced[p, s] += actual_produce
                step_production += actual_produce
                self.wip[p, s] -= actual_produce
                if s < self.num_procs - 1:
                    self.wip[p, s+1] += actual_produce
                    
                # 계획 달성률
                plan_qty = self.plan[p, s]
                cum_produced = self.produced[p, s]
                achievement_rate = (cum_produced / plan_qty * 100.0) if plan_qty > 0 else 0.0
                
                # BATCH_ID 매핑 (배치 정보가 없으면 N/A)
                batch_id = 'N/A'
                batch_df = self.data.get('batch_tool_info', pd.DataFrame())
                if not batch_df.empty:
                    match = batch_df[(batch_df['PLAN_PROD_KEY'] == self.products[p]) & (batch_df['OPER_ID'] == self.processes[s])]
                    if not match.empty:
                        batch_id = match.iloc[0]['BATCH_ID']
                    
                if plan_qty > 0 or actual_produce > 0 or active_count > 0:
                    self.production_logs.append({
                        'TIME_SLOT': self.current_step,
                        'BATCH_ID': batch_id,
                        'PLAN_PROD_KEY': self.products[p],
                        'OPER_ID': self.processes[s],
                        'ACTIVE_EQP_CNT': active_count,
                        'TRANSITION_EQP_CNT': transition_count,
                        'PRODUCTION_QTY': actual_produce,
                        'REMAIN_WIP': self.wip[p, s],
                        'UTILIZATION_RATE(%)': round(utilization, 2),
                        'CUM_PRODUCED': cum_produced,
                        'PLAN_QTY': plan_qty,
                        'ACHIEVEMENT_RATE(%)': round(achievement_rate, 2)
                    })
                    
        # Resolve target eqp instantly for this mock
        self.active_eqp += self.target_eqp
        self.target_eqp.fill(0)
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        # Reward calculation (Incremental)
        # 1. 이번 시간(Step)에 생산한 양에 비례한 보상
        total_plan = np.sum(self.plan) + 1e-6
        reward = step_production / total_plan
        
        # 2. 장비 이동이 발생한 경우 페널티 부여 (불필요한 이동 억제)
        if len(transfers) > 0:
            reward -= 0.01 * len(transfers)
            
        # 3. 에피소드 종료 시 최종 계획 달성률에 따른 보너스 부여
        if terminated:
            final_achievement = np.sum(self.produced) / total_plan
            if final_achievement >= 0.99:
                reward += 1.0 # 목표 달성 시 큰 보너스
        
        return self._get_obs(), float(reward), terminated, False, {'transfers': transfers}
