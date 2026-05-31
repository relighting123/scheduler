"""Input 스냅샷 7종 → 정적 배치 최적화 문제 정의."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

DAY_BOUNDARY_HOUR = 7


def parse_plan_time(time_val) -> Optional[datetime]:
    s = str(time_val).strip()
    if len(s) < 8:
        return None
    s = s.ljust(10, "0")[:10]
    try:
        hour = int(s[8:10])
    except ValueError:
        return None
    if hour == 24:
        return datetime.strptime(s[:8], "%Y%m%d") + timedelta(days=1)
    if hour > 23:
        return None
    try:
        return datetime.strptime(s, "%Y%m%d%H")
    except ValueError:
        return None


def plan_segment_hours(start_time: str, end_time: str) -> float:
    start_dt = parse_plan_time(start_time)
    end_dt = parse_plan_time(end_time)
    if start_dt is None or end_dt is None:
        return 24.0
    delta = (end_dt - start_dt).total_seconds() / 3600.0
    return max(delta, 1.0)


@dataclass(frozen=True)
class SlotKey:
    plan_prod_key: str
    oper_id: str

    def as_tuple(self) -> Tuple[str, str]:
        return (self.plan_prod_key, self.oper_id)


@dataclass
class OperSlot:
    """단일 (계획제품, 공정) 슬롯."""

    key: SlotKey
    oper_seq: int
    batch_id: str
    wip_qty: float
    plan_qty: float
    horizon_hours: float
    uph_by_model: Dict[str, float] = field(default_factory=dict)
    allocation: Dict[str, int] = field(default_factory=dict)

    def capacity_per_hour(self) -> float:
        return sum(
            self.allocation.get(model, 0) * self.uph_by_model.get(model, 0.0)
            for model in self.uph_by_model
        )


@dataclass
class PlanAllocationProblem:
    """Input 7종으로 구성된 정적 최적화 문제."""

    slots: List[OperSlot]
    model_pool: Dict[str, int]
    tool_limit: Dict[Tuple[str, str], int]  # (batch_id, model) → max eqp
    slot_by_key: Dict[Tuple[str, str], OperSlot] = field(default_factory=dict)

    def sorted_slots(self) -> List[OperSlot]:
        return sorted(self.slots, key=lambda s: (s.key.plan_prod_key, s.oper_seq))


def _is_available(avail_df: pd.DataFrame, prod: str, oper: str, model: str) -> bool:
    if avail_df.empty:
        return True
    rows = avail_df[
        (avail_df["PLAN_PROD_KEY"] == prod)
        & (avail_df["OPER_ID"] == oper)
        & (avail_df["EQP_MODEL_CD"] == model)
    ]
    if rows.empty:
        return True
    return str(rows.iloc[0].get("AVAIL_YN", "Y")).strip().upper() != "N"


def build_problem(data: Dict[str, pd.DataFrame]) -> PlanAllocationProblem:
    wip_df = data.get("wip_info", pd.DataFrame())
    uph_df = data.get("uph_info", pd.DataFrame())
    eqp_df = data.get("eqp_qty_info", pd.DataFrame())
    avail_df = data.get("avail_info", pd.DataFrame())
    batch_df = data.get("batch_tool_info", pd.DataFrame())
    tool_df = data.get("tool_qty_info", pd.DataFrame())
    plan_df = data.get("plan_info", pd.DataFrame())

    batch_map: Dict[Tuple[str, str], str] = {}
    if not batch_df.empty:
        for _, row in batch_df.iterrows():
            batch_map[(str(row["PLAN_PROD_KEY"]), str(row["OPER_ID"]))] = str(row["BATCH_ID"])

    wip_map: Dict[Tuple[str, str], Tuple[float, int]] = {}
    if not wip_df.empty:
        for _, row in wip_df.iterrows():
            key = (str(row["PLAN_PROD_KEY"]), str(row["OPER_ID"]))
            seq = int(row.get("OPER_SEQ", 0) or 0)
            wip_map[key] = (float(row.get("WIP_QTY", 0) or 0), seq)

    plan_qty_map: Dict[Tuple[str, str], float] = {}
    plan_hours_map: Dict[Tuple[str, str], float] = {}
    if not plan_df.empty:
        for _, row in plan_df.iterrows():
            key = (str(row["PLAN_PROD_KEY"]), str(row["OPER_ID"]))
            plan_qty_map[key] = plan_qty_map.get(key, 0.0) + float(row.get("PLAN_QTY", 0) or 0)
            seg_h = plan_segment_hours(row.get("START_TIME", ""), row.get("END_TIME", ""))
            plan_hours_map[key] = max(plan_hours_map.get(key, 0.0), seg_h)

    uph_map: Dict[Tuple[str, str], Dict[str, float]] = {}
    if not uph_df.empty:
        for _, row in uph_df.iterrows():
            prod, oper, model = (
                str(row["PLAN_PROD_KEY"]),
                str(row["OPER_ID"]),
                str(row["EQP_MODEL_CD"]),
            )
            uph = float(row.get("UPH", 0) or 0)
            if uph <= 0 or not _is_available(avail_df, prod, oper, model):
                continue
            uph_map.setdefault((prod, oper), {})[model] = uph

    current_alloc: Dict[Tuple[str, str], Dict[str, int]] = {}
    if not eqp_df.empty and not batch_df.empty:
        merged = pd.merge(batch_df, eqp_df, on="BATCH_ID", how="inner")
        for _, row in merged.iterrows():
            key = (str(row["PLAN_PROD_KEY"]), str(row["OPER_ID"]))
            model = str(row["EQP_MODEL_CD"])
            qty = int(float(row.get("EQP_QTY", 0) or 0))
            current_alloc.setdefault(key, {})
            current_alloc[key][model] = current_alloc[key].get(model, 0) + qty

    model_pool: Dict[str, int] = {}
    if not eqp_df.empty:
        grouped = eqp_df.groupby("EQP_MODEL_CD")["EQP_QTY"].sum()
        for model, qty in grouped.items():
            model_pool[str(model)] = int(qty)

    tool_limit: Dict[Tuple[str, str], int] = {}
    if not tool_df.empty:
        for _, row in tool_df.iterrows():
            tool_limit[(str(row["BATCH_ID"]), str(row["EQP_MODEL_CD"]))] = int(
                float(row.get("TOOL_QTY", 0) or 0)
            )

    all_keys = set(wip_map) | set(plan_qty_map) | set(uph_map) | set(current_alloc)
    slots: List[OperSlot] = []
    slot_by_key: Dict[Tuple[str, str], OperSlot] = {}

    for key in sorted(all_keys):
        prod, oper = key
        wip, seq = wip_map.get(key, (0.0, 0))
        batch_id = batch_map.get(key, f"{prod}_{oper}")
        models = uph_map.get(key, {})
        if not models:
            continue
        slot = OperSlot(
            key=SlotKey(prod, oper),
            oper_seq=seq,
            batch_id=batch_id,
            wip_qty=wip,
            plan_qty=plan_qty_map.get(key, 0.0),
            horizon_hours=plan_hours_map.get(key, 24.0),
            uph_by_model=models,
            allocation=dict(current_alloc.get(key, {})),
        )
        for model in models:
            slot.allocation.setdefault(model, 0)
        slots.append(slot)
        slot_by_key[key] = slot

    return PlanAllocationProblem(
        slots=slots,
        model_pool=model_pool,
        tool_limit=tool_limit,
        slot_by_key=slot_by_key,
    )
