"""장비 1대 이동·추가 액션 (정적 최적화·RL 공통)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

from biz.services.plan_allocation.models import PlanAllocationProblem

MoveType = Literal["noop", "add", "transfer"]


@dataclass(frozen=True)
class AllocationMove:
    kind: MoveType
    src_idx: int = -1
    dst_idx: int = -1
    model: str = ""

    def describe(self, problem: PlanAllocationProblem) -> dict:
        if self.kind == "noop":
            return {"type": "noop"}
        if self.kind == "add":
            slot = problem.slots[self.dst_idx]
            return {
                "type": "add",
                "model": self.model,
                "to": slot.key.as_tuple(),
                "qty": 1,
            }
        src = problem.slots[self.src_idx]
        dst = problem.slots[self.dst_idx]
        return {
            "type": "transfer",
            "model": self.model,
            "from": src.key.as_tuple(),
            "to": dst.key.as_tuple(),
            "qty": 1,
        }


def total_assigned(problem: PlanAllocationProblem) -> dict:
    totals = {m: 0 for m in problem.model_pool}
    for slot in problem.slots:
        for model, cnt in slot.allocation.items():
            totals[model] = totals.get(model, 0) + cnt
    return totals


def tool_ok(problem: PlanAllocationProblem, slot, model: str, new_qty: int) -> bool:
    limit = problem.tool_limit.get((slot.batch_id, model))
    if limit is None:
        return True
    return new_qty <= limit


def pool_ok(problem: PlanAllocationProblem, model: str, delta: int = 1) -> bool:
    totals = total_assigned(problem)
    return totals.get(model, 0) + delta <= problem.model_pool.get(model, 0)


def apply_transfer(problem, src_idx: int, dst_idx: int, model: str) -> bool:
    src = problem.slots[src_idx]
    dst = problem.slots[dst_idx]
    if src.allocation.get(model, 0) <= 0:
        return False
    new_dst = dst.allocation.get(model, 0) + 1
    if not tool_ok(problem, dst, model, new_dst):
        return False
    src.allocation[model] -= 1
    dst.allocation[model] = new_dst
    return True


def apply_add(problem, dst_idx: int, model: str) -> bool:
    slot = problem.slots[dst_idx]
    if model not in slot.uph_by_model:
        return False
    if not pool_ok(problem, model, 1):
        return False
    new_q = slot.allocation.get(model, 0) + 1
    if not tool_ok(problem, slot, model, new_q):
        return False
    slot.allocation[model] = new_q
    return True


def enumerate_moves(problem: PlanAllocationProblem) -> List[AllocationMove]:
    moves: List[AllocationMove] = [AllocationMove("noop")]
    totals = total_assigned(problem)

    for di, dst in enumerate(problem.slots):
        for model in dst.uph_by_model:
            if dst.uph_by_model[model] <= 0:
                continue
            pool = problem.model_pool.get(model, 0)
            if totals.get(model, 0) >= pool:
                continue
            new_q = dst.allocation.get(model, 0) + 1
            if tool_ok(problem, dst, model, new_q):
                moves.append(AllocationMove("add", dst_idx=di, model=model))

    for si, src in enumerate(problem.slots):
        for di, dst in enumerate(problem.slots):
            if si == di:
                continue
            for model in set(src.allocation) | set(dst.uph_by_model):
                if src.allocation.get(model, 0) <= 0:
                    continue
                if model not in dst.uph_by_model:
                    continue
                new_dst = dst.allocation.get(model, 0) + 1
                if tool_ok(problem, dst, model, new_dst):
                    moves.append(
                        AllocationMove("transfer", src_idx=si, dst_idx=di, model=model)
                    )
    return moves


def apply_move(problem: PlanAllocationProblem, move: AllocationMove) -> bool:
    if move.kind == "noop":
        return True
    if move.kind == "add":
        return apply_add(problem, move.dst_idx, move.model)
    return apply_transfer(problem, move.src_idx, move.dst_idx, move.model)
