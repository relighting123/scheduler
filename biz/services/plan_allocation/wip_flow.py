"""공정별 재공 밸런스 — 선행 공정 WIP 과다 시 우선 배치."""

from __future__ import annotations

from typing import Dict, List, Tuple

from biz.services.plan_allocation.models import OperSlot, PlanAllocationProblem
from biz.services.plan_allocation.static_capacity import _group_slots_by_product


def wip_cover_hours(slot: OperSlot) -> float:
    """현재 배치 기준 재공을 소진하는 데 걸리는 시간(시간). 용량 0이면 inf."""
    cap_h = slot.capacity_per_hour()
    if cap_h <= 0:
        return float("inf")
    return slot.wip_qty / cap_h


def compute_flow_priorities(problem: PlanAllocationProblem) -> Dict[Tuple[str, str], float]:
    """슬롯별 장비 배치 우선순위 (1.0 근처 = 먼저 붙여서 흐름을 맞출 대상).

    아이디어:
    - 공정 순서대로 WIP 커버 시간(재공/시간당용량)이 크면 병목·적체 후보
    - 앞 공정 커버가 뒷 공정보다 훨씬 크면 앞 공정 우선 (재공 밸런스 교정)
    - 용량이 없거나 계획이 없으면 우선순위 낮춤
    """
    priorities: Dict[Tuple[str, str], float] = {}

    for prod, chain in _group_slots_by_product(problem).items():
        if not chain:
            continue
        covers = [wip_cover_hours(s) for s in chain]
        finite = [c for c in covers if c < float("inf")]
        max_cover = max(finite) if finite else 1.0

        for i, slot in enumerate(chain):
            key = slot.key.as_tuple()
            cover = covers[i]
            if cover == float("inf") or max_cover <= 0:
                priorities[key] = 0.1
                continue

            # 절대 적체: 커버 시간이 길수록 우선
            pile = min(cover / max(max_cover, 1.0), 2.0)

            # 상대 적체: 앞 공정 vs 바로 다음 공정
            balance = 1.0
            if i + 1 < len(chain):
                next_cover = covers[i + 1]
                if next_cover < float("inf") and next_cover > 0:
                    balance = min(cover / next_cover, 3.0)
                elif cover > 0:
                    balance = 2.0

            # 선행 공정일수록 가중 (동일 cover면 OP10 쪽 우선)
            seq_weight = 1.0 + 0.15 * max(len(chain) - 1 - i, 0)

            raw = pile * balance * seq_weight
            priorities[key] = min(max(raw, 0.1), 3.0)

    return priorities


def flow_balance_penalty(problem: PlanAllocationProblem) -> float:
    """제품×공정 체인 내 WIP 커버 시간 편차 — 클수록 밸런스 불량."""
    penalty = 0.0
    for _prod, chain in _group_slots_by_product(problem).items():
        covers = [wip_cover_hours(s) for s in chain]
        finite = [c for c in covers if c < float("inf")]
        if len(finite) < 2:
            continue
        mean_c = sum(finite) / len(finite)
        var = sum((c - mean_c) ** 2 for c in finite) / len(finite)
        penalty += var**0.5
    return penalty


def describe_flow_balance(problem: PlanAllocationProblem) -> List[dict]:
    """리포트용: 공정별 재공·커버시간·우선순위."""
    pri = compute_flow_priorities(problem)
    rows = []
    for slot in problem.sorted_slots():
        key = slot.key.as_tuple()
        cap = slot.capacity_per_hour()
        rows.append(
            {
                "PLAN_PROD_KEY": slot.key.plan_prod_key,
                "OPER_ID": slot.key.oper_id,
                "OPER_SEQ": slot.oper_seq,
                "WIP_QTY": round(slot.wip_qty, 1),
                "CAPACITY_PER_HOUR": round(cap, 2),
                "WIP_COVER_HOURS": round(wip_cover_hours(slot), 2)
                if cap > 0
                else None,
                "FLOW_PRIORITY": round(pri.get(key, 1.0), 3),
            }
        )
    rows.sort(key=lambda r: (-r["FLOW_PRIORITY"], r["PLAN_PROD_KEY"], r["OPER_SEQ"]))
    return rows
