"""콘솔·JSON용 계획 달성 기여 리포트."""

from __future__ import annotations

from typing import Any, Dict

from biz.services.plan_allocation.optimizer import OptimizationResult


def format_allocation_report(result: OptimizationResult) -> str:
    lines = [
        "",
        "=" * 72,
        " [계획 기반 장비 배치 분석] (Input-only · 시뮬레이터 미사용)",
        "=" * 72,
        f"  초기 전체 달성률(정적): {result.initial_summary.overall_achievement:.2f}%",
        f"  최적화 후:           {result.optimized_summary.overall_achievement:.2f}%",
        f"  평균(공정별):        {result.optimized_summary.avg_achievement:.2f}%",
        f"  탐색 이동 횟수:      {result.iterations}",
        "",
        "── 제품별 마지막 공정 달성률(정적 추정) ──",
    ]
    for prod, rate in sorted(result.optimized_summary.by_product_last_oper.items()):
        lines.append(f"  {prod}: {rate:.2f}%")

    lines.append("")
    lines.append("── (계획제품, 공정)별 기여 ──")
    for slot in result.optimized_summary.slot_contributions:
        alloc = ", ".join(f"{m}×{q}" for m, q in slot.allocation.items() if q > 0)
        lines.append(
            f"  {slot.plan_prod_key}/{slot.oper_id}: "
            f"계획 {slot.plan_qty:.0f} → 달성가능 {slot.achievable_qty:.0f} "
            f"({slot.achievement_rate:.1f}%) | "
            f"용량 {slot.capacity_per_hour:.0f}/h | 병목 {slot.binding} | 배치 [{alloc or '-'}]"
        )

    lines.append("")
    lines.append("── 모델별 풀 활용 ──")
    for model, util in sorted(result.optimized_summary.model_utilization.items()):
        lines.append(
            f"  {model}: {util['assigned']}/{util['pool']}대 "
            f"(가동 {util['utilization(%)']:.1f}%)"
        )

    if result.moves:
        lines.append("")
        lines.append("── 권장 재배치 이동 ──")
        for i, mv in enumerate(result.moves[:20], 1):
            if mv["type"] == "transfer":
                lines.append(
                    f"  {i}. {mv['model']} 1대: {mv['from']} → {mv['to']}"
                )
            else:
                lines.append(f"  {i}. {mv['model']} 1대 추가 → {mv['to']}")

    if result.recommended_allocation:
        lines.append("")
        lines.append("── 권장 배치 (target_allocation 형식) ──")
        for prod, opers in sorted(result.recommended_allocation.items()):
            for oper, models in sorted(opers.items()):
                m_str = ", ".join(f"{m}:{q}" for m, q in sorted(models.items()))
                lines.append(f"  {prod}/{oper}: {m_str}")

    lines.append("=" * 72)
    return "\n".join(lines)


def result_as_json(result: OptimizationResult, marginal: list = None) -> Dict[str, Any]:
    payload = result.to_dict()
    if marginal is not None:
        payload["marginal_contribution"] = marginal
    return payload
