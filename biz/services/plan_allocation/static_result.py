"""정적 배치 결과 (시간대 없음) — 표·JSON·CSV."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from biz.services.plan_allocation.optimizer import OptimizationResult

STATIC_ALLOC_COLUMNS = [
    "PLAN_PROD_KEY",
    "OPER_ID",
    "EQP_MODEL_CD",
    "EQP_QTY",
]


def build_static_allocation_rows(
    result: OptimizationResult,
) -> List[Dict[str, Any]]:
    """계획제품×공정×장비모델별 권장 대수 (flat)."""
    rows: List[Dict[str, Any]] = []
    for prod, opers in sorted(result.recommended_allocation.items()):
        for oper, models in sorted(opers.items()):
            for model, qty in sorted(models.items()):
                if int(qty) <= 0:
                    continue
                rows.append(
                    {
                        "PLAN_PROD_KEY": prod,
                        "OPER_ID": oper,
                        "EQP_MODEL_CD": model,
                        "EQP_QTY": int(qty),
                    }
                )
    return rows


def build_static_allocation_nested(result: OptimizationResult) -> Dict[str, Any]:
    """ground_truth target_allocation 과 동일한 중첩 구조."""
    return dict(result.recommended_allocation)


def static_allocation_payload(
    result: OptimizationResult,
    rule_timekey: Optional[str] = None,
) -> Dict[str, Any]:
    rows = build_static_allocation_rows(result)
    summary = result.optimized_summary
    return {
        "type": "static_equipment_allocation",
        "rule_timekey": rule_timekey,
        "allocation": build_static_allocation_nested(result),
        "allocation_table": rows,
        "achievement": {
            "overall_percent": summary.overall_achievement,
            "avg_by_oper_percent": summary.avg_achievement,
            "last_oper_by_product_percent": dict(summary.by_product_last_oper),
        },
    }


def format_static_allocation_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "(배치 결과 없음)"
    df = pd.DataFrame(rows, columns=STATIC_ALLOC_COLUMNS)
    lines = [
        "",
        "=" * 72,
        " [정적 장비 배치 결과] 계획제품 × 공정 × 장비모델 → 대수",
        " (시간대·시뮬레이터·강화학습 없음 — Input 스냅샷 기준)",
        "=" * 72,
        df.to_string(index=False),
        "=" * 72,
    ]
    return "\n".join(lines)


def save_static_allocation_files(
    payload: Dict[str, Any],
    output_dir: str = "output",
    basename: str = "static_allocation",
) -> Dict[str, str]:
    """JSON + CSV 저장. 저장 경로 dict 반환."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tk = payload.get("rule_timekey") or "snapshot"
    safe_tk = "".join(c if c.isalnum() else "_" for c in str(tk))
    stem = f"{basename}_{safe_tk}"

    json_path = out / f"{stem}.json"
    csv_path = out / f"{stem}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    table = payload.get("allocation_table") or []
    pd.DataFrame(table, columns=STATIC_ALLOC_COLUMNS).to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )

    return {"json": str(json_path), "csv": str(csv_path)}
