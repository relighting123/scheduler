import numpy as np

class HeuristicExpert:
    def __init__(self, env):
        self.env = env
        self.num_prods = env.num_prods
        self.num_procs = env.num_procs
        self.num_models = env.num_models

    def select_action(self):
        """환경 상태(WIP, ST 등)를 분석하여 가장 최적의 장비 이동 액션을 반환합니다."""
        wip = self.env.wip
        
        priorities = np.zeros((self.num_prods, self.num_procs))
        
        # 1. 병목 공정 파악 (우선순위 계산)
        for p in range(self.num_prods):
            for s in range(self.num_procs):
                feasible_allocation = 0.0
                for m in range(self.num_models):
                    if self.env.st_matrix[p, s, m] > 0:
                        feasible_allocation += self.env.active_eqp[p, s, m] + self.env.target_eqp[p, s, m]
                
                sts = self.env.st_matrix[p, s, :]
                positive_sts = sts[sts > 0]
                min_st = positive_sts.min() if positive_sts.size > 0 else 999999
                
                if min_st < 999999:
                    workload_hours = (wip[p, s] * min_st) / 60.0
                    priority = workload_hours / (feasible_allocation + 1.0)
                    
                    if s == self.num_procs - 1 and wip[p, s] > 0:
                        priority *= 2.0  # 최종 공정에 남은 물량은 빠르게 처리하도록 우선순위 증가
                    priorities[p, s] = priority

        best_flat_idx = np.argmax(priorities)
        best_p, best_s = best_flat_idx // self.num_procs, best_flat_idx % self.num_procs
        max_priority = priorities[best_p, best_s]
        
        # 2. 가장 덜 바쁜(여유로운) 장비 수배
        potential_sources = []
        
        # IDLE 장비 우선 탐색
        for m in range(self.num_models):
            if self.env.idle_eqp[m] > 0 and self.env.st_matrix[best_p, best_s, m] > 0:
                potential_sources.append(('IDLE', 'IDLE', m, -999.0)) # IDLE 장비는 무조건 1순위 타겟
        
        # 타 공정의 장비 탐색
        for p in range(self.num_prods):
            for s in range(self.num_procs):
                for m in range(self.num_models):
                    if self.env.active_eqp[p, s, m] > 0 and self.env.st_matrix[best_p, best_s, m] > 0:
                        st_val = self.env.st_matrix[p, s, m]
                        if st_val > 0:
                            source_feasible_alloc = sum(self.env.active_eqp[p, s, :] + self.env.target_eqp[p, s, :])
                            current_val = (wip[p, s] * st_val / 60.0) / (source_feasible_alloc + 1e-6)
                            potential_sources.append((p, s, m, current_val))
                        else:
                            potential_sources.append((p, s, m, -1.0))
        
        if not potential_sources:
            return 0 # 이동 가능한 장비 없음
            
        MOVE_THRESHOLD = 2.0
        MIN_WORK_TO_MOVE = 3.0
        
        # 가장 여유로운 장비(priority 값이 가장 작은 장비)를 찾음
        best_source = min(potential_sources, key=lambda x: x[3])
        src_p, src_s, src_m, src_priority = best_source
        
        sts = self.env.st_matrix[best_p, best_s, :]
        positive_sts = sts[sts > 0]
        min_st_target = positive_sts.min() if positive_sts.size > 0 else 999999
        
        target_workload = (wip[best_p, best_s] * min_st_target) / 60.0
        
        # 3. 이동 결정 및 Action 반환
        # IDLE 장비이거나, (우선순위 차이가 임계값 이상 && 목표 지점의 남은 작업량이 충분한 경우)
        if src_priority == -999.0 or (max_priority > src_priority + MOVE_THRESHOLD and target_workload >= MIN_WORK_TO_MOVE):
            if best_p == src_p and best_s == src_s:
                return 0
            
            # New Action Format: target_idx = prod * (procs * models) + proc * models + model
            target_idx = best_p * (self.env.num_procs * self.env.num_models) + best_s * self.env.num_models + src_m
            return target_idx + 1
            
        return 0
