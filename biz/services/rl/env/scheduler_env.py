import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict
from datetime import datetime
import os

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
        self.total_eqp_hours = np.zeros((self.num_prods, self.num_procs))
        self.operating_eqp_hours = np.zeros((self.num_prods, self.num_procs))

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
        
        # Precompute batch_id mapping to speed up step()
        self.batch_id_map = {}
        if not batch_df.empty:
            for _, row in batch_df.iterrows():
                self.batch_id_map[(row['PLAN_PROD_KEY'], row['OPER_ID'])] = row['BATCH_ID']
        
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
        
        # 초반(current_step == 0)에는 장비 전환 생성이 있을 수 없으므로 action을 0(전환 없음)으로 처리
        if self.current_step == 0:
            action = 0
            
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
                    
                # If no IDLE, pull from another active node (Priority-based: pull from least needed node first)
                if not moved:
                    best_src_p, best_src_s = None, None
                    min_priority = float('inf')
                    
                    for p in range(self.num_prods):
                        for s in range(self.num_procs):
                            if self.active_eqp[p, s, t_model] > 0 and (p != t_prod or s != t_proc):
                                uph = 0.0
                                st_val = self.st_matrix[p, s, t_model]
                                if st_val > 0.0:
                                    uph = 60.0 / st_val
                                    
                                # Priority: Lower priority means the equipment is less needed here.
                                # If UPH is 0, this equipment model cannot process this product at all, so priority is -1.0 (highest priority to pull).
                                if uph == 0.0:
                                    priority = -1.0
                                else:
                                    priority = self.wip[p, s] / uph
                                    
                                if priority < min_priority:
                                    min_priority = priority
                                    best_src_p, best_src_s = p, s
                                    
                    if best_src_p is not None:
                        self.active_eqp[best_src_p, best_src_s, t_model] -= 1
                        self.target_eqp[t_prod, t_proc, t_model] += 1
                        moved = True
                        transfers.append({
                            'FROM_PROD': self.products[best_src_p], 'FROM_PROC': self.processes[best_src_s],
                            'TO_PROD': self.products[t_prod], 'TO_PROC': self.processes[t_proc],
                            'MODEL': self.models[t_model]
                        })
                    
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
                
                # 가동/유휴 시간 누적 (장비 대수 * 시간)
                self.total_eqp_hours[p, s] += active_count * 1.0
                self.operating_eqp_hours[p, s] += (actual_produce / capacity * active_count * 1.0) if capacity > 0 else 0.0
                
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
                batch_id = self.batch_id_map.get((self.products[p], self.processes[s]), 'N/A')
                    
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
        
        # 2. 장비 이동이 발생한 경우 페널티 부여 (패널티를 0.01에서 0.002로 완화하여 장비 재배치를 유도)
        if len(transfers) > 0:
            reward -= 0.002 * len(transfers)
            
        # 3. 에피소드 종료 시 최종 계획 달성률에 따른 보너스 부여 (계단식 보상 추가로 Sparse Reward 개선)
        if terminated:
            final_achievement = np.sum(self.produced) / total_plan
            if final_achievement >= 0.99:
                reward += 1.5
            elif final_achievement >= 0.90:
                reward += 0.8
            elif final_achievement >= 0.80:
                reward += 0.4
            elif final_achievement >= 0.60:
                reward += 0.1
        
        return self._get_obs(), float(reward), terminated, False, {'transfers': transfers}

    def print_final_summary(self, method_name: str = "simulation"):
        """최종 기준에서 제품별 공정별 장비 할당대수와 계획 달성률/가동률 정보를 출력하고 엑셀 파일로 저장합니다."""
        print("\n" + "="*100)
        print(f" [최종 장비 할당 및 실적/가동률 요약 리포트 - {method_name.upper()}]")
        print("="*100)
        
        summary_data = []
        
        # Header formatting: 방법||장비||모델1||모델2||.. ||계획||실적||달성률||전체시간||가동시간||유휴시간||가동률
        headers = ["방법", "장비"] + self.models + ["계획", "실적", "달성률(%)", "전체시간(Hr)", "가동시간(Hr)", "유휴시간(Hr)", "가동률(%)"]
        header_line = "||".join(headers)
        print(header_line)
        print("-" * len(header_line))
        
        for p in range(self.num_prods):
            p_name = self.products[p]
            # Skip padding products
            if "PAD_PROD" in p_name:
                continue
            for s in range(self.num_procs):
                s_name = self.processes[s]
                # Skip padding processes
                if "PAD_PROC" in s_name:
                    continue
                
                # Check if there is any plan, production, or active equipment
                plan_qty = self.plan[p, s]
                produced_qty = self.produced[p, s]
                active_eqp_list = [self.active_eqp[p, s, m] for m in range(self.num_models)]
                total_active = sum(active_eqp_list)
                
                if plan_qty == 0 and produced_qty == 0 and total_active == 0:
                    continue
                
                row_key = f"{p_name}_{s_name}"
                
                # Equipment model counts
                eqp_counts = [int(count) for count in active_eqp_list] # Store as int for Excel
                
                # Calculations
                achievement = (produced_qty / plan_qty * 100.0) if plan_qty > 0 else 0.0
                total_hr = self.total_eqp_hours[p, s]
                op_hr = self.operating_eqp_hours[p, s]
                idle_hr = max(0.0, total_hr - op_hr)
                util_rate = (op_hr / total_hr * 100.0) if total_hr > 0 else 0.0
                
                row_values_for_print = (
                    [method_name, row_key] + 
                    [f"{count}" for count in eqp_counts] + # Convert back to string for printing
                    [
                        f"{plan_qty:,.1f}", 
                        f"{produced_qty:,.1f}", 
                        f"{achievement:.2f}%", 
                        f"{total_hr:.1f}", 
                        f"{op_hr:.1f}", 
                        f"{idle_hr:.1f}", 
                        f"{util_rate:.2f}%"
                    ]
                )
                print("||".join(row_values_for_print))

                # Collect data for Excel
                summary_row = {
                    "방법": method_name,
                    "장비": row_key,
                }
                for i, model_name in enumerate(self.models):
                    summary_row[model_name] = eqp_counts[i]
                summary_row.update({
                    "계획": plan_qty,
                    "실적": produced_qty,
                    "달성률(%)": round(achievement, 2),
                    "전체시간(Hr)": round(total_hr, 1),
                    "가동시간(Hr)": round(op_hr, 1),
                    "유휴시간(Hr)": round(idle_hr, 1),
                    "가동률(%)": round(util_rate, 2)
                })
                summary_data.append(summary_row)

        print("="*100 + "\n")

        # Save to Excel
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            df_logs = pd.DataFrame()
            if hasattr(self, 'production_logs') and self.production_logs:
                df_logs = pd.DataFrame(self.production_logs)
                if not df_logs.empty:
                    df_logs.rename(columns={
                        'TIME_SLOT': '시간대',
                        'BATCH_ID': '배치ID',
                        'PLAN_PROD_KEY': '제품',
                        'OPER_ID': '공정',
                        'ACTIVE_EQP_CNT': '가동장비대수',
                        'TRANSITION_EQP_CNT': '전이장비대수',
                        'PRODUCTION_QTY': '시간당생산량',
                        'REMAIN_WIP': '남은WIP',
                        'UTILIZATION_RATE(%)': '가동률(%)',
                        'CUM_PRODUCED': '누적생산량',
                        'PLAN_QTY': '계획량',
                        'ACHIEVEMENT_RATE(%)': '달성률(%)'
                    }, inplace=True)
                    df_logs.insert(0, "방법", method_name)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "logs/simulation_logs"
            import os
            os.makedirs(output_dir, exist_ok=True)
            excel_filename = os.path.join(output_dir, f"final_summary_{method_name}_{timestamp}.xlsx")
            
            # Write sheets
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                df_summary.to_excel(writer, sheet_name="최종 요약", index=False)
                if not df_logs.empty:
                    df_logs.to_excel(writer, sheet_name="시간대별 상세", index=False)
                    
            # Apply styling
            try:
                import openpyxl
                from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
                from openpyxl.utils import get_column_letter
                
                wb = openpyxl.load_workbook(excel_filename)
                
                # Colors and styles
                header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
                header_font = Font(name="Malgun Gothic", size=11, bold=True, color="FFFFFF")
                regular_font = Font(name="Malgun Gothic", size=10)
                
                thin_side = Side(border_style="thin", color="D9D9D9")
                thin_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
                
                for sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                    
                    # Columns styling and sizing
                    for col_idx, col in enumerate(ws.columns, 1):
                        max_len = 0
                        for cell in col:
                            cell.font = regular_font
                            cell.border = thin_border
                            cell.alignment = Alignment(horizontal="center", vertical="center")
                            
                            if isinstance(cell.value, (int, float)):
                                header_val = ws.cell(row=1, column=col_idx).value
                                if header_val and ("달성률" in str(header_val) or "가동률" in str(header_val)):
                                    cell.number_format = '#,##0.0"%"'
                                else:
                                    cell.number_format = '#,##0'
                                    
                            val_str = str(cell.value or '')
                            if len(val_str) > max_len:
                                max_len = len(val_str)
                                
                        col_letter = get_column_letter(col_idx)
                        ws.column_dimensions[col_letter].width = max(max_len + 4, 12)
                        
                    # Style headers
                    for cell in ws[1]:
                        cell.fill = header_fill
                        cell.font = header_font
                        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                    ws.row_dimensions[1].height = 28
                    
                wb.save(excel_filename)
            except Exception as e:
                print(f"엑셀 스타일 적용 중 오류 발생: {e}")
                
            print(f"요약 및 시간대별 리포트가 '{excel_filename}' 파일로 저장되었습니다.")
