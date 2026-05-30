"""시뮬레이션 종료 후 장비 할당·계획달성률 DataFrame 생성 (추론·벤치마크 공통)."""

import re

import numpy as np
import pandas as pd


def build_final_allocation_df(env):
    """제품·공정·모델별 최종 장비 할당 대수."""
    rows = []
    for p_idx, prod in enumerate(env.products):
        if prod.startswith("PAD_PROD_") or prod.startswith("_EMPTY"):
            continue
        for s_idx, oper in enumerate(env.processes):
            if oper.startswith("PAD_PROC_") or oper.startswith("_EMPTY"):
                continue
            for m_idx, model in enumerate(env.models):
                qty = float(env.active_eqp[p_idx, s_idx, m_idx])
                if qty <= 0:
                    continue
                rounded_qty = int(round(qty)) if np.isclose(qty, round(qty)) else round(qty, 4)
                rows.append({
                    "PLAN_PROD_KEY": prod,
                    "OPER_ID": oper,
                    "EQP_MODEL_CD": model,
                    "ALLOCATED_EQP_QTY": rounded_qty,
                })

    allocation_df = pd.DataFrame(
        rows,
        columns=["PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD", "ALLOCATED_EQP_QTY"],
    )
    if not allocation_df.empty:
        allocation_df = allocation_df.sort_values(
            by=["PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD"]
        ).reset_index(drop=True)
    return allocation_df


def build_last_process_achievement_df(env, data):
    """마지막 공정 기준 제품별 계획 달성률."""
    last_oper_by_prod = {}
    wip_df = data.get("wip_info", pd.DataFrame())

    if not wip_df.empty and {"PLAN_PROD_KEY", "OPER_ID", "OPER_SEQ"}.issubset(wip_df.columns):
        tmp = wip_df.copy()
        tmp["OPER_SEQ_NUM"] = pd.to_numeric(tmp["OPER_SEQ"], errors="coerce")
        tmp = tmp.sort_values(by=["PLAN_PROD_KEY", "OPER_SEQ_NUM", "OPER_ID"])
        last_rows = tmp.groupby("PLAN_PROD_KEY", as_index=False).tail(1)
        for _, row in last_rows.iterrows():
            last_oper_by_prod[row["PLAN_PROD_KEY"]] = row["OPER_ID"]

    process_candidates = [proc for proc in env.processes if not proc.startswith("PAD_PROC_")]
    products = [
        prod for prod in env.products
        if not prod.startswith("PAD_PROD_") and not prod.startswith("_EMPTY")
    ]
    for prod in products:
        if prod in last_oper_by_prod or not process_candidates:
            continue
        fallback_oper = max(
            process_candidates,
            key=lambda oper: int(re.search(r"\d+", oper).group()) if re.search(r"\d+", oper) else -1,
        )
        last_oper_by_prod[prod] = fallback_oper

    rows = []
    for prod in products:
        oper = last_oper_by_prod.get(prod)
        if oper is None:
            continue
        p_idx = env.prod_idx.get(prod)
        s_idx = env.proc_idx.get(oper)
        if p_idx is None or s_idx is None:
            continue

        produced_qty = float(env.produced[p_idx, s_idx])
        plan_qty = float(env.plan[p_idx, s_idx])
        rate = (produced_qty / plan_qty * 100.0) if plan_qty > 0 else 0.0

        rows.append({
            "PLAN_PROD_KEY": prod,
            "LAST_OPER_ID": oper,
            "PRODUCED_QTY": round(produced_qty, 4),
            "PLAN_QTY": round(plan_qty, 4),
            "ACHIEVEMENT_RATE(%)": round(rate, 2),
        })

    achievement_df = pd.DataFrame(
        rows,
        columns=["PLAN_PROD_KEY", "LAST_OPER_ID", "PRODUCED_QTY", "PLAN_QTY", "ACHIEVEMENT_RATE(%)"],
    )
    if not achievement_df.empty:
        achievement_df = achievement_df.sort_values(by=["PLAN_PROD_KEY"]).reset_index(drop=True)
    return achievement_df


def build_allocation_pivot_df(allocation_df):
    """장비 모델별 할당 대수 피벗."""
    if allocation_df.empty:
        return pd.DataFrame(columns=["PLAN_PROD_KEY", "OPER_ID"])

    pivot_df = allocation_df.pivot_table(
        index=["PLAN_PROD_KEY", "OPER_ID"],
        columns="EQP_MODEL_CD",
        values="ALLOCATED_EQP_QTY",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    pivot_df.columns.name = None

    model_cols = [col for col in pivot_df.columns if col not in ["PLAN_PROD_KEY", "OPER_ID"]]
    for col in model_cols:
        numeric_values = pd.to_numeric(pivot_df[col], errors="coerce").fillna(0.0)
        if np.all(np.isclose(numeric_values, np.round(numeric_values))):
            pivot_df[col] = np.round(numeric_values).astype(int)
        else:
            pivot_df[col] = numeric_values.round(4)

    return pivot_df.sort_values(by=["PLAN_PROD_KEY", "OPER_ID"]).reset_index(drop=True)
