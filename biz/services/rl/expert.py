import numpy as np
from typing import Any, Dict, Optional


def apply_target_allocation(env, allocation: Dict[str, Any]) -> np.ndarray:
    """ground_truth target_allocation → env 차원 목표 행렬."""
    target = np.zeros((env.num_prods, env.num_procs, env.num_models))
    if not allocation:
        return target
    for prod, opers in allocation.items():
        if prod not in env.prod_idx:
            continue
        for oper, models in opers.items():
            if oper not in env.proc_idx:
                continue
            for model, qty in models.items():
                if model in env.model_idx:
                    target[env.prod_idx[prod], env.proc_idx[oper], env.model_idx[model]] = float(qty)
    return target


def default_benchmark_target_allocation() -> Dict[str, Any]:
    """기존 benchmark_dataset 정답 배치."""
    return {
        "P1": {"OP10": {"MODEL_A": 5}, "OP20": {"MODEL_A": 5}},
        "P2": {"OP10": {"MODEL_B": 3}, "OP20": {"MODEL_B": 3}},
        "P3": {
            "OP10": {"MODEL_C": 2, "MODEL_B": 1},
            "OP20": {"MODEL_C": 3, "MODEL_B": 1},
        },
    }


class HeuristicExpert:
    def __init__(self, env):
        self.env = env
        self.num_prods = env.num_prods
        self.num_procs = env.num_procs
        self.num_models = env.num_models

    def select_action(self):
        """환경 상태(WIP, ST 등)를 분석하여 가장 최적의 장비 이동 액션을 반환합니다."""
        if self.env.current_step == 0:
            return 0

        wip = self.env.wip

        priorities = np.zeros((self.num_prods, self.num_procs))

        for p in range(self.num_prods):
            for s in range(self.num_procs):
                feasible_allocation = 0.0
                for m in range(self.num_models):
                    if self.env.st_matrix[p, s, m] > 0:
                        feasible_allocation += (
                            self.env.active_eqp[p, s, m] + self.env.target_eqp[p, s, m]
                        )

                sts = self.env.st_matrix[p, s, :]
                positive_sts = sts[sts > 0]
                min_st = positive_sts.min() if positive_sts.size > 0 else 999999

                if min_st < 999999:
                    workload_hours = (wip[p, s] * min_st) / 60.0
                    priority = workload_hours / (feasible_allocation + 1.0)

                    if s == self.num_procs - 1 and wip[p, s] > 0:
                        priority *= 2.0
                    priorities[p, s] = priority

        best_flat_idx = np.argmax(priorities)
        best_p, best_s = best_flat_idx // self.num_procs, best_flat_idx % self.num_procs
        max_priority = priorities[best_p, best_s]

        potential_sources = []

        for m in range(self.num_models):
            if self.env.idle_eqp[m] > 0 and self.env.st_matrix[best_p, best_s, m] > 0:
                potential_sources.append(("IDLE", "IDLE", m, -999.0))

        for p in range(self.num_prods):
            for s in range(self.num_procs):
                for m in range(self.num_models):
                    if self.env.active_eqp[p, s, m] > 0 and self.env.st_matrix[best_p, best_s, m] > 0:
                        st_val = self.env.st_matrix[p, s, m]
                        if st_val > 0:
                            source_feasible_alloc = sum(
                                self.env.active_eqp[p, s, :] + self.env.target_eqp[p, s, :]
                            )
                            current_val = (wip[p, s] * st_val / 60.0) / (source_feasible_alloc + 1e-6)
                            potential_sources.append((p, s, m, current_val))
                        else:
                            potential_sources.append((p, s, m, -1.0))

        if not potential_sources:
            return 0

        MOVE_THRESHOLD = 2.0
        MIN_WORK_TO_MOVE = 3.0

        best_source = min(potential_sources, key=lambda x: x[3])
        src_p, src_s, src_m, src_priority = best_source

        sts = self.env.st_matrix[best_p, best_s, :]
        positive_sts = sts[sts > 0]
        min_st_target = positive_sts.min() if positive_sts.size > 0 else 999999

        target_workload = (wip[best_p, best_s] * min_st_target) / 60.0

        if src_priority == -999.0 or (
            max_priority > src_priority + MOVE_THRESHOLD and target_workload >= MIN_WORK_TO_MOVE
        ):
            if best_p == src_p and best_s == src_s:
                return 0

            target_idx = (
                best_p * (self.env.num_procs * self.env.num_models)
                + best_s * self.env.num_models
                + src_m
            )
            return target_idx + 1

        return 0


class OptimalExpert:
    def __init__(self, env, target_allocation: Optional[Dict[str, Any]] = None):
        self.env = env
        self.num_prods = env.num_prods
        self.num_procs = env.num_procs
        self.num_models = env.num_models

        alloc = target_allocation or default_benchmark_target_allocation()
        self.target = apply_target_allocation(env, alloc)

    def select_action(self):
        """Finds the next equipment move to match the target optimal allocation."""
        for p in range(self.num_prods):
            for s in range(self.num_procs):
                for m in range(self.num_models):
                    curr = self.env.active_eqp[p, s, m] + self.env.target_eqp[p, s, m]
                    tgt = self.target[p, s, m]
                    if curr < tgt:
                        for sp in range(self.num_prods):
                            for ss in range(self.num_procs):
                                scurr = self.env.active_eqp[sp, ss, m]
                                stgt = self.target[sp, ss, m]
                                if scurr > stgt:
                                    target_idx = (
                                        p * (self.env.num_procs * self.env.num_models)
                                        + s * self.env.num_models
                                        + m
                                    )
                                    return target_idx + 1
        return 0
