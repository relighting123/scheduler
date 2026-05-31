"""Input-only 조합 탐색(이웃 이동) 장비 배치 최적화."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

from biz.services.plan_allocation.models import PlanAllocationProblem, build_problem
from biz.services.plan_allocation.static_capacity import (
    PlanAchievementSummary,
    evaluate_allocation,
    marginal_gain_if_add,
)


@dataclass
class OptimizationResult:
    problem: PlanAllocationProblem
    initial_summary: PlanAchievementSummary
    optimized_summary: PlanAchievementSummary
    recommended_allocation: Dict[str, Dict[str, Dict[str, int]]]
    moves: List[dict]
    iterations: int

    def to_dict(self) -> dict:
        return {
            "initial": self.initial_summary.to_dict(),
            "optimized": self.optimized_summary.to_dict(),
            "recommended_allocation": self.recommended_allocation,
            "moves": self.moves,
            "iterations": self.iterations,
        }


def _allocation_to_nested(problem: PlanAllocationProblem) -> Dict[str, Dict[str, Dict[str, int]]]:
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for slot in problem.slots:
        prod = slot.key.plan_prod_key
        oper = slot.key.oper_id
        alloc = {m: q for m, q in slot.allocation.items() if q > 0}
        if not alloc:
            continue
        out.setdefault(prod, {})[oper] = alloc
    return out


def _total_assigned(problem: PlanAllocationProblem) -> Dict[str, int]:
    totals: Dict[str, int] = {m: 0 for m in problem.model_pool}
    for slot in problem.slots:
        for model, cnt in slot.allocation.items():
            totals[model] = totals.get(model, 0) + cnt
    return totals


def _tool_ok(problem: PlanAllocationProblem, slot, model: str, new_qty: int) -> bool:
    limit = problem.tool_limit.get((slot.batch_id, model))
    if limit is None:
        return True
    return new_qty <= limit


def _pool_ok(problem: PlanAllocationProblem, model: str, delta: int) -> bool:
    totals = _total_assigned(problem)
    return totals.get(model, 0) + delta <= problem.model_pool.get(model, 0)


def _objective(summary: PlanAchievementSummary, last_oper_weight: float = 1.5) -> float:
    """전체 계획 달성 + 마지막 공정 가중."""
    base = summary.overall_achievement
    if not summary.by_product_last_oper:
        return base
    last_avg = sum(summary.by_product_last_oper.values()) / len(summary.by_product_last_oper)
    return base + (last_oper_weight - 1.0) * last_avg


class PlanAllocationOptimizer:
    """시뮬레이터 없이 Input만으로 장비 대수 재배치를 탐색한다."""

    def __init__(
        self,
        data: Dict,
        max_iterations: int = 200,
        last_oper_weight: float = 1.5,
    ):
        self.data = data
        self.max_iterations = max_iterations
        self.last_oper_weight = last_oper_weight

    def run(self, optimize: bool = True) -> OptimizationResult:
        problem = build_problem(self.data)
        initial = evaluate_allocation(problem)
        moves: List[dict] = []
        iterations = 0

        if optimize:
            candidates = [
                copy.deepcopy(problem),
                self._greedy_from_pool(build_problem(self.data)),
            ]
            best_working = problem
            best_summary = initial
            best_moves: List[dict] = []

            for working in candidates:
                trial_moves: List[dict] = []
                it = self._hill_climb(working, trial_moves)
                summary = evaluate_allocation(working)
                if _objective(summary, self.last_oper_weight) > _objective(
                    best_summary, self.last_oper_weight
                ):
                    best_working = working
                    best_summary = summary
                    best_moves = trial_moves
                    iterations = it

            optimized = best_summary
            return OptimizationResult(
                problem=best_working,
                initial_summary=initial,
                optimized_summary=optimized,
                recommended_allocation=_allocation_to_nested(best_working),
                moves=best_moves,
                iterations=iterations,
            )

        return OptimizationResult(
            problem=problem,
            initial_summary=initial,
            optimized_summary=initial,
            recommended_allocation=_allocation_to_nested(problem),
            moves=[],
            iterations=0,
        )

    def _hill_climb(self, problem: PlanAllocationProblem, moves: List[dict]) -> int:
        iterations = 0
        while iterations < self.max_iterations:
            current_obj = _objective(
                evaluate_allocation(problem), self.last_oper_weight
            )
            best_move = None
            best_obj = current_obj

            # 같은 모델 1대: 슬롯 A → 슬롯 B 이동
            for src in problem.slots:
                for dst in problem.slots:
                    if src.key == dst.key:
                        continue
                    for model in set(src.allocation) | set(dst.uph_by_model):
                        if src.allocation.get(model, 0) <= 0:
                            continue
                        if model not in dst.uph_by_model:
                            continue
                        if not _try_transfer(problem, src, dst, model):
                            continue
                        trial_obj = _objective(
                            evaluate_allocation(problem), self.last_oper_weight
                        )
                        _undo_transfer(problem, src, dst, model)
                        if trial_obj > best_obj + 1e-6:
                            best_obj = trial_obj
                            best_move = ("transfer", src, dst, model, trial_obj)

            # 풀 잔여 1대: 미배치 슬롯에 추가
            totals = _total_assigned(problem)
            for slot in problem.slots:
                for model, uph in slot.uph_by_model.items():
                    if uph <= 0:
                        continue
                    pool = problem.model_pool.get(model, 0)
                    if totals.get(model, 0) >= pool:
                        continue
                    new_q = slot.allocation.get(model, 0) + 1
                    if not _tool_ok(problem, slot, model, new_q):
                        continue
                    slot.allocation[model] = new_q
                    trial_obj = _objective(
                        evaluate_allocation(problem), self.last_oper_weight
                    )
                    slot.allocation[model] = new_q - 1
                    if trial_obj > best_obj + 1e-6:
                        best_obj = trial_obj
                        best_move = ("add", slot, model, trial_obj)

            if best_move is None:
                break

            if best_move[0] == "transfer":
                _, src, dst, model, _ = best_move
                _try_transfer(problem, src, dst, model)
                moves.append(
                    {
                        "type": "transfer",
                        "model": model,
                        "from": src.key.as_tuple(),
                        "to": dst.key.as_tuple(),
                        "qty": 1,
                    }
                )
            else:
                _, slot, model, _ = best_move
                slot.allocation[model] = slot.allocation.get(model, 0) + 1
                moves.append(
                    {
                        "type": "add",
                        "model": model,
                        "to": slot.key.as_tuple(),
                        "qty": 1,
                    }
                )
            iterations += 1

        return iterations

    def _greedy_from_pool(self, problem: PlanAllocationProblem) -> PlanAllocationProblem:
        """모델 풀을 한 대씩 가장 달성률 기여가 큰 슬롯에 배치한다."""
        working = copy.deepcopy(problem)
        for slot in working.slots:
            slot.allocation = {m: 0 for m in slot.uph_by_model}

        for model, pool in working.model_pool.items():
            for _ in range(pool):
                best_slot = None
                best_gain = -1.0
                for slot in working.slots:
                    if model not in slot.uph_by_model:
                        continue
                    key = slot.key.as_tuple()
                    new_q = slot.allocation.get(model, 0) + 1
                    if not _tool_ok(working, slot, model, new_q):
                        continue
                    if not _pool_ok(working, model, 1):
                        continue
                    gain = marginal_gain_if_add(working, key, model)
                    if gain > best_gain:
                        best_gain = gain
                        best_slot = slot
                if best_slot is None:
                    break
                best_slot.allocation[model] = best_slot.allocation.get(model, 0) + 1
        return working

    def marginal_report(self) -> List[dict]:
        """현재 배치 기준 +1대 시 공정별 기여도(정적)."""
        problem = build_problem(self.data)
        rows = []
        for slot in problem.slots:
            key = slot.key.as_tuple()
            for model in slot.uph_by_model:
                gain = marginal_gain_if_add(problem, key, model)
                cap_delta = slot.uph_by_model[model]
                rows.append(
                    {
                        "PLAN_PROD_KEY": slot.key.plan_prod_key,
                        "OPER_ID": slot.key.oper_id,
                        "EQP_MODEL_CD": model,
                        "CURRENT_EQP": slot.allocation.get(model, 0),
                        "UPH_PER_EQP": cap_delta,
                        "MARGINAL_OVERALL_ACHIEVEMENT(%)": round(gain, 4),
                        "EXTRA_CAPACITY_PER_HOUR": cap_delta,
                    }
                )
        rows.sort(key=lambda r: r["MARGINAL_OVERALL_ACHIEVEMENT(%)"], reverse=True)
        return rows


def _try_transfer(problem, src, dst, model: str) -> bool:
    if src.allocation.get(model, 0) <= 0:
        return False
    new_dst = dst.allocation.get(model, 0) + 1
    if not _tool_ok(problem, dst, model, new_dst):
        return False
    src.allocation[model] -= 1
    dst.allocation[model] = new_dst
    return True


def _undo_transfer(problem, src, dst, model: str) -> None:
    src.allocation[model] = src.allocation.get(model, 0) + 1
    dst.allocation[model] = max(dst.allocation.get(model, 1) - 1, 0)
