"""벤치마크 시뮬레이션 결과를 통합 리포트(초기·최종 장비대수, 생산/계획, 달성률, 가동률)로 집계합니다."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def capture_initial_allocation(env) -> np.ndarray:
    """reset 직후(최초 시점) 제품×공정×모델별 장비 대수 스냅샷."""
    return np.array(env.active_eqp, copy=True)


def _is_real_product(name: str) -> bool:
    return not (name.startswith("PAD_PROD_") or name.startswith("_EMPTY"))


def _is_real_process(name: str) -> bool:
    return not (name.startswith("PAD_PROC_") or name.startswith("_EMPTY"))


def build_scenario_detail_df(
    env,
    initial_eqp: np.ndarray,
    method_name: str,
) -> Tuple[pd.DataFrame, float]:
    """
    제품·공정별 리포트:
      - 최초(초기) 장비모델별 대수 → 최종 장비모델별 대수
      - 생산량·계획량·달성률·장비가동률(%)
    반환: (상세 DataFrame, 전체 평균 장비가동률 %)
    """
    rows: List[Dict[str, Any]] = []
    util_rates: List[float] = []
    util_weights: List[float] = []

    for p in range(env.num_prods):
        p_name = env.products[p]
        if not _is_real_product(p_name):
            continue
        for s in range(env.num_procs):
            s_name = env.processes[s]
            if not _is_real_process(s_name):
                continue

            plan_qty = float(env.plan[p, s])
            produced_qty = float(env.produced[p, s])
            init_counts = [int(round(initial_eqp[p, s, m])) for m in range(env.num_models)]
            final_counts = [int(round(env.active_eqp[p, s, m])) for m in range(env.num_models)]

            if plan_qty == 0 and produced_qty == 0 and sum(init_counts) == 0 and sum(final_counts) == 0:
                continue

            achievement = (produced_qty / plan_qty * 100.0) if plan_qty > 0 else 0.0
            total_hr = float(env.total_eqp_hours[p, s])
            op_hr = float(env.operating_eqp_hours[p, s])
            util_rate = (op_hr / total_hr * 100.0) if total_hr > 0 else 0.0

            row: Dict[str, Any] = {
                "방법": method_name,
                "제품": p_name,
                "공정": s_name,
            }
            for m_idx, model_name in enumerate(env.models):
                row[f"초기_{model_name}"] = init_counts[m_idx]
                row[f"최종_{model_name}"] = final_counts[m_idx]
            row.update({
                "생산량": round(produced_qty, 2),
                "계획량": round(plan_qty, 2),
                "달성률(%)": round(achievement, 2),
                "장비가동률(%)": round(util_rate, 2),
            })
            rows.append(row)

            if total_hr > 0:
                util_rates.append(util_rate)
                util_weights.append(total_hr)

    detail_df = pd.DataFrame(rows)
    if util_rates:
        avg_util = round(
            float(np.average(util_rates, weights=util_weights)),
            2,
        )
    else:
        avg_util = 0.0

    return detail_df, avg_util


def print_scenario_detail_report(
    scenario_id: str,
    method_label: str,
    detail_df: pd.DataFrame,
    avg_utilization_pct: float,
) -> None:
    """콘솔에 벤치마크 상세 리포트 출력."""
    print("\n" + "-" * 100)
    print(f" [벤치마크 상세 — {scenario_id} / {method_label}]")
    print(f" 평균 장비가동률: {avg_utilization_pct}%")
    print("-" * 100)
    if detail_df.empty:
        print(" (집계할 데이터 없음)")
    else:
        print(detail_df.to_string(index=False))
    print("-" * 100)


def build_comparison_row(
    label: str,
    op20_metrics: Dict[str, Any],
    avg_utilization_pct: float,
) -> Dict[str, Any]:
    return {
        "Method": label,
        "P1 달성률(%)": op20_metrics.get("P1_OP20_ACHIEVEMENT", 0),
        "P2 달성률(%)": op20_metrics.get("P2_OP20_ACHIEVEMENT", 0),
        "P3 달성률(%)": op20_metrics.get("P3_OP20_ACHIEVEMENT", 0),
        "평균 달성률(%)": op20_metrics.get("AVG_ACHIEVEMENT", 0),
        "평균 장비가동률(%)": avg_utilization_pct,
        "장비 전환 횟수": op20_metrics.get("TRANSFERS", 0),
    }
