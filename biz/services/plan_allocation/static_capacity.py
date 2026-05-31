"""시뮬레이터 없이 Input 기반 정적 생산·달성률 추정."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from biz.services.plan_allocation.models import OperSlot, PlanAllocationProblem


@dataclass
class SlotContribution:
    plan_prod_key: str
    oper_id: str
    oper_seq: int
    plan_qty: float
    wip_qty: float
    horizon_hours: float
    allocation: Dict[str, int]
    capacity_per_hour: float
    max_producible: float
    achievable_qty: float
    achievement_rate: float
    binding: str  # WIP | CAPACITY | PLAN | UPSTREAM
    upstream_feed: float = 0.0

    def to_dict(self) -> dict:
        return {
            "PLAN_PROD_KEY": self.plan_prod_key,
            "OPER_ID": self.oper_id,
            "OPER_SEQ": self.oper_seq,
            "PLAN_QTY": round(self.plan_qty, 2),
            "WIP_QTY": round(self.wip_qty, 2),
            "HORIZON_HOURS": round(self.horizon_hours, 2),
            "ALLOCATION": dict(self.allocation),
            "CAPACITY_PER_HOUR": round(self.capacity_per_hour, 2),
            "MAX_PRODUCIBLE": round(self.max_producible, 2),
            "ACHIEVABLE_QTY": round(self.achievable_qty, 2),
            "ACHIEVEMENT_RATE(%)": round(self.achievement_rate, 2),
            "BINDING": self.binding,
            "UPSTREAM_FEED": round(self.upstream_feed, 2),
        }


@dataclass
class PlanAchievementSummary:
    slot_contributions: List[SlotContribution] = field(default_factory=list)
    by_product_last_oper: Dict[str, float] = field(default_factory=dict)
    avg_achievement: float = 0.0
    total_plan_qty: float = 0.0
    total_achievable: float = 0.0
    overall_achievement: float = 0.0
    model_utilization: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "slots": [s.to_dict() for s in self.slot_contributions],
            "by_product_last_oper": {
                k: round(v, 2) for k, v in self.by_product_last_oper.items()
            },
            "avg_achievement": round(self.avg_achievement, 2),
            "total_plan_qty": round(self.total_plan_qty, 2),
            "total_achievable": round(self.total_achievable, 2),
            "overall_achievement": round(self.overall_achievement, 2),
            "model_utilization": self.model_utilization,
        }


def _group_slots_by_product(problem: PlanAllocationProblem) -> Dict[str, List[OperSlot]]:
    groups: Dict[str, List[OperSlot]] = {}
    for slot in problem.sorted_slots():
        groups.setdefault(slot.key.plan_prod_key, []).append(slot)
    for prod in groups:
        groups[prod].sort(key=lambda s: s.oper_seq)
    return groups


def evaluate_allocation(problem: PlanAllocationProblem) -> PlanAchievementSummary:
    """공정 순서·WIP·용량 상한으로 정적 달성 가능량을 계산한다."""
    contributions: List[SlotContribution] = []
    last_oper_by_product: Dict[str, float] = {}
    achievement_rates: List[float] = []
    total_plan = 0.0
    total_achievable = 0.0

    for prod, chain in _group_slots_by_product(problem).items():
        upstream_out = 0.0
        last_rate = 0.0
        for slot in chain:
            cap_h = slot.capacity_per_hour()
            cap_total = cap_h * slot.horizon_hours
            available_wip = slot.wip_qty + upstream_out
            max_prod = min(cap_total, available_wip) if cap_total > 0 else 0.0

            if slot.plan_qty > 0:
                achievable = min(max_prod, slot.plan_qty)
            else:
                achievable = max_prod

            if slot.plan_qty > 0:
                rate = achievable / slot.plan_qty * 100.0
            else:
                rate = 100.0 if achievable > 0 else 0.0

            if cap_total <= 0:
                binding = "NO_CAPACITY"
            elif max_prod <= 0:
                binding = "NO_WIP"
            elif achievable >= slot.plan_qty and slot.plan_qty > 0:
                binding = "PLAN"
            elif max_prod < cap_total and max_prod <= available_wip:
                binding = "WIP" if slot.wip_qty + upstream_out <= cap_total else "CAPACITY"
            else:
                binding = "CAPACITY"

            if upstream_out > 0 and binding == "WIP" and slot.wip_qty < max_prod:
                binding = "UPSTREAM"

            contributions.append(
                SlotContribution(
                    plan_prod_key=slot.key.plan_prod_key,
                    oper_id=slot.key.oper_id,
                    oper_seq=slot.oper_seq,
                    plan_qty=slot.plan_qty,
                    wip_qty=slot.wip_qty,
                    horizon_hours=slot.horizon_hours,
                    allocation=dict(slot.allocation),
                    capacity_per_hour=cap_h,
                    max_producible=max_prod,
                    achievable_qty=achievable,
                    achievement_rate=rate,
                    binding=binding,
                    upstream_feed=upstream_out,
                )
            )
            if slot.plan_qty > 0:
                achievement_rates.append(rate)
                total_plan += slot.plan_qty
                total_achievable += achievable

            upstream_out = achievable
            last_rate = rate
        last_oper_by_product[prod] = last_rate

    model_used: Dict[str, int] = {m: 0 for m in problem.model_pool}
    for slot in problem.slots:
        for model, cnt in slot.allocation.items():
            model_used[model] = model_used.get(model, 0) + cnt

    model_util = {}
    for model, pool in problem.model_pool.items():
        used = model_used.get(model, 0)
        model_util[model] = {
            "pool": pool,
            "assigned": used,
            "idle": max(pool - used, 0),
            "utilization(%)": round((used / pool * 100.0) if pool > 0 else 0.0, 2),
        }

    avg = sum(achievement_rates) / len(achievement_rates) if achievement_rates else 0.0
    overall = (total_achievable / total_plan * 100.0) if total_plan > 0 else 0.0

    return PlanAchievementSummary(
        slot_contributions=contributions,
        by_product_last_oper=last_oper_by_product,
        avg_achievement=avg,
        total_plan_qty=total_plan,
        total_achievable=total_achievable,
        overall_achievement=overall,
        model_utilization=model_util,
    )


def marginal_gain_if_add(
    problem: PlanAllocationProblem,
    slot_key: Tuple[str, str],
    model: str,
) -> float:
    """해당 슬롯에 모델 1대 추가 시 전체 달성률(가중) 개선량."""
    before = evaluate_allocation(problem).overall_achievement
    slot = problem.slot_by_key[slot_key]
    slot.allocation[model] = slot.allocation.get(model, 0) + 1
    after = evaluate_allocation(problem).overall_achievement
    slot.allocation[model] = max(slot.allocation.get(model, 1) - 1, 0)
    return after - before
