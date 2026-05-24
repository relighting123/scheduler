"""학습 스냅샷 간 고정 obs/action 차원을 위한 엔티티 스키마 유틸."""
from typing import Dict, List, Optional, Tuple

import pandas as pd

TABLE_KEYS = (
    "wip_info",
    "uph_info",
    "eqp_qty_info",
    "avail_info",
    "batch_tool_info",
    "tool_qty_info",
    "plan_info",
)


def discover_entities_from_data(
    data: Dict[str, pd.DataFrame],
    max_prods: Optional[int] = None,
    max_procs: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """단일 스냅샷에서 제품·공정·모델 목록 추출 (패딩 없음)."""
    prods, procs, models = set(), set(), set()
    wip_df = data.get("wip_info", pd.DataFrame())
    uph_df = data.get("uph_info", pd.DataFrame())

    if not wip_df.empty:
        if "PLAN_PROD_KEY" in wip_df.columns:
            prods.update(wip_df["PLAN_PROD_KEY"].astype(str).unique())
        if "OPER_ID" in wip_df.columns:
            procs.update(wip_df["OPER_ID"].astype(str).unique())

    if not uph_df.empty:
        if "PLAN_PROD_KEY" in uph_df.columns:
            prods.update(uph_df["PLAN_PROD_KEY"].astype(str).unique())
        if "OPER_ID" in uph_df.columns:
            procs.update(uph_df["OPER_ID"].astype(str).unique())
        if "EQP_MODEL_CD" in uph_df.columns:
            models.update(uph_df["EQP_MODEL_CD"].astype(str).unique())

    products = sorted(prods)
    processes = sorted(procs)
    model_list = sorted(models)

    if max_prods is not None:
        products = products[:max_prods]
    if max_procs is not None:
        processes = processes[:max_procs]

    if not products:
        products = ["_EMPTY_PROD_"]
    if not processes:
        processes = ["_EMPTY_PROC_"]
    if not model_list:
        model_list = ["_EMPTY_MODEL_"]

    return products, processes, model_list


def compute_canonical_schema(
    snapshots: List[Dict[str, pd.DataFrame]],
    max_prods: Optional[int] = None,
    max_procs: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """여러 스냅샷의 합집합으로 고정 학습 차원 정의."""
    all_prods, all_procs, all_models = set(), set(), set()
    for data in snapshots:
        p, s, m = discover_entities_from_data(data, max_prods=max_prods, max_procs=max_procs)
        all_prods.update(p)
        all_procs.update(s)
        all_models.update(m)

    products = sorted(all_prods)
    processes = sorted(all_procs)
    models = sorted(all_models)

    if max_prods is not None:
        products = products[:max_prods]
    if max_procs is not None:
        processes = processes[:max_procs]

    if not products:
        products = ["_EMPTY_PROD_"]
    if not processes:
        processes = ["_EMPTY_PROC_"]
    if not models:
        models = ["_EMPTY_MODEL_"]

    return products, processes, models
