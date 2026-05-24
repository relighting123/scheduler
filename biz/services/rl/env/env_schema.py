"""학습 스냅샷 간 고정 obs/action 차원을 위한 엔티티 스키마 유틸."""
import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
from gymnasium import spaces

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


def obs_dim_from_canonical(
    products: List[str],
    processes: List[str],
) -> int:
    """SchedulerEnv._get_obs() 차원 (num_prods * num_procs * 8 + 2)."""
    return len(products) * len(processes) * 8 + 2


def action_dim_from_canonical(
    products: List[str],
    processes: List[str],
    models: List[str],
) -> int:
    return len(products) * len(processes) * len(models) + 1


def build_spaces_from_canonical(
    products: List[str],
    processes: List[str],
    models: List[str],
) -> Tuple[spaces.Box, spaces.Discrete]:
    obs_dim = obs_dim_from_canonical(products, processes)
    return (
        spaces.Box(low=0, high=1000, shape=(obs_dim,), dtype=np.float32),
        spaces.Discrete(action_dim_from_canonical(products, processes, models)),
    )


def schema_dict(
    products: List[str],
    processes: List[str],
    models: List[str],
) -> Dict[str, List[str]]:
    return {
        "products": list(products),
        "processes": list(processes),
        "models": list(models),
    }


def save_env_schema(
    products: List[str],
    processes: List[str],
    models: List[str],
    model_path: str = "scheduler_ppo_model",
) -> str:
    path = f"{model_path}.schema.json"
    payload = schema_dict(products, processes, models)
    payload["obs_dim"] = obs_dim_from_canonical(products, processes)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_env_schema(model_path: str = "scheduler_ppo_model") -> Optional[Dict[str, List[str]]]:
    path = f"{model_path}.schema.json"
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
