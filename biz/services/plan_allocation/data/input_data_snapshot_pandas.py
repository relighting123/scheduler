"""RTS_LINEDSDB_INF EAV → env 스냅샷 변환 (pandas 폴백).

DB 조회 시에는 input_data_snapshot_sql.fetch_snapshot_from_db()가
필터·집계·수치 변환을 SQL에서 수행한다.
본 모듈의 transform_input_data_snapshot()은 EAV DataFrame이 이미 메모리에 있을 때만 사용한다.
"""

from typing import Dict

import pandas as pd

from biz.services.plan_allocation.data.input_data_constants import (
    GBN_ASSIGN_EQUIP,
    GBN_D0_TARGET,
    GBN_D1_TARGET,
    GBN_TOOL,
    GBN_UPH,
    GBN_WIP,
    INPUT_DATA_COLUMNS,
    empty_snapshot_frames,
    plan_windows_for_rule_timekey,
)


def _numeric_attr(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce").fillna(0.0)


def transform_input_data_snapshot(
    df: pd.DataFrame,
    rule_timekey: str,
) -> Dict[str, pd.DataFrame]:
    """단일 RULE_TIMEKEY에 대한 RTS_LINEDSDB_INF 행을 env 입력 7종 DataFrame으로 변환."""
    empty = empty_snapshot_frames()
    if df is None or df.empty:
        return empty

    snap = df.copy()
    for col in INPUT_DATA_COLUMNS:
        if col not in snap.columns:
            snap[col] = ""
    snap["GBN_CD"] = snap["GBN_CD"].astype(str).str.strip().str.upper()
    snap["ATTR_VAL"] = snap["ATTR_VAL"].astype(str).str.strip()

    d0_window, d1_window = plan_windows_for_rule_timekey(rule_timekey)
    time_slot = str(rule_timekey).strip()[:10]

    # WIP
    wip_rows = snap[snap["GBN_CD"] == GBN_WIP]
    wip_info = pd.DataFrame(columns=["PLAN_PROD_KEY", "OPER_ID", "OPER_SEQ", "WIP_QTY"])
    if not wip_rows.empty:
        wip_info = (
            wip_rows.groupby(["PLAN_PROD_KEY", "OPER_ID"], as_index=False)
            .agg(OPER_SEQ=("OPER_SEQ", "max"), WIP_QTY=("ATTR_VAL", "first"))
        )
        wip_info["WIP_QTY"] = _numeric_attr(wip_info["WIP_QTY"])

    # UPH
    uph_rows = snap[snap["GBN_CD"] == GBN_UPH]
    uph_info = pd.DataFrame(columns=["PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD", "UPH"])
    if not uph_rows.empty:
        uph_info = uph_rows[["PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD", "ATTR_VAL"]].copy()
        uph_info = uph_info.rename(columns={"ATTR_VAL": "UPH"})
        uph_info["UPH"] = _numeric_attr(uph_info["UPH"])
        uph_info = uph_info[uph_info["UPH"] > 0].drop_duplicates(
            subset=["PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD"]
        )

    # 설비 대수
    assign_rows = snap[snap["GBN_CD"] == GBN_ASSIGN_EQUIP]
    eqp_qty_info = pd.DataFrame(columns=["BATCH_ID", "EQP_MODEL_CD", "TIME_SLOT", "EQP_QTY"])
    if not assign_rows.empty:
        eqp_qty_info = assign_rows[["BATCH_ID", "EQP_MODEL_CD", "ATTR_VAL"]].copy()
        eqp_qty_info["TIME_SLOT"] = time_slot
        eqp_qty_info = eqp_qty_info.rename(columns={"ATTR_VAL": "EQP_QTY"})
        eqp_qty_info["EQP_QTY"] = _numeric_attr(eqp_qty_info["EQP_QTY"])

    # Tool 수량
    tool_rows = snap[snap["GBN_CD"] == GBN_TOOL]
    tool_qty_info = pd.DataFrame(columns=["BATCH_ID", "EQP_MODEL_CD", "TOOL_QTY"])
    if not tool_rows.empty:
        tool_qty_info = tool_rows[["BATCH_ID", "EQP_MODEL_CD", "ATTR_VAL"]].copy()
        tool_qty_info = tool_qty_info.rename(columns={"ATTR_VAL": "TOOL_QTY"})
        tool_qty_info["TOOL_QTY"] = _numeric_attr(tool_qty_info["TOOL_QTY"])

    # Batch–제품–공정 매핑 (측정값이 있는 행 기준)
    dim_cols = ["BATCH_ID", "PLAN_PROD_KEY", "OPER_ID"]
    batch_tool_info = (
        snap[snap["BATCH_ID"].astype(str).str.strip() != ""][dim_cols]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if batch_tool_info.empty:
        batch_tool_info = empty["batch_tool_info"]

    # UPH 유무로 처리 가능 여부 (UPH 없으면 진행 불가)
    avail_info = pd.DataFrame(columns=["PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD", "AVAIL_YN"])
    if not uph_info.empty:
        avail_info = uph_info[["PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD"]].copy()
        avail_info["AVAIL_YN"] = "Y"
    else:
        uph_candidates = snap[
            (snap["PLAN_PROD_KEY"].astype(str).str.strip() != "")
            & (snap["OPER_ID"].astype(str).str.strip() != "")
            & (snap["EQP_MODEL_CD"].astype(str).str.strip() != "")
        ][["PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD"]].drop_duplicates()
        if not uph_candidates.empty:
            avail_info = uph_candidates.copy()
            avail_info["AVAIL_YN"] = "N"

    # D0/D1 계획
    plan_parts = []
    for gbn, window in ((GBN_D0_TARGET, d0_window), (GBN_D1_TARGET, d1_window)):
        target_rows = snap[snap["GBN_CD"] == gbn]
        if target_rows.empty:
            continue
        part = target_rows[["PLAN_PROD_KEY", "OPER_ID", "ATTR_VAL"]].copy()
        part["START_TIME"] = window[0]
        part["END_TIME"] = window[1]
        part = part.rename(columns={"ATTR_VAL": "PLAN_QTY"})
        part["PLAN_QTY"] = _numeric_attr(part["PLAN_QTY"])
        plan_parts.append(part)
    plan_info = (
        pd.concat(plan_parts, ignore_index=True)
        if plan_parts
        else empty["plan_info"]
    )

    return {
        "wip_info": wip_info if not wip_info.empty else empty["wip_info"],
        "uph_info": uph_info if not uph_info.empty else empty["uph_info"],
        "eqp_qty_info": eqp_qty_info if not eqp_qty_info.empty else empty["eqp_qty_info"],
        "avail_info": avail_info if not avail_info.empty else empty["avail_info"],
        "batch_tool_info": batch_tool_info,
        "tool_qty_info": tool_qty_info if not tool_qty_info.empty else empty["tool_qty_info"],
        "plan_info": plan_info if not plan_info.empty else empty["plan_info"],
    }
