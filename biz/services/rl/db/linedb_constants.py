"""RTS_LINEDSDB_INF EAV 테이블 메타·GBN 코드·스냅샷 빈 프레임·D0/D1 계획 구간."""

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd

LINEDB_TABLE = "RTS_LINEDSDB_INF"
DAY_BOUNDARY_HOUR = 7
LINEDB_COLUMNS = [
    "RULE_TIMEKEY",
    "FAC_ID",
    "BATCH_ID",
    "PLAN_PROD_KEY",
    "OPER_ID",
    "OPER_SEQ",
    "EQP_MODEL_CD",
    "GBN_CD",
    "ATTR_VAL",
]

GBN_WIP = "WIP_QTY"
GBN_UPH = "UPH"
GBN_ASSIGN_EQUIP = "ASSIGN_EQUIP_CNT"
GBN_D0_TARGET = "D0_TARGET_QTY"
GBN_D1_TARGET = "D1_TARGET_QTY"
GBN_TOOL = "TOOL_QTY"


def parse_rule_timekey(rule_timekey: str) -> Optional[datetime]:
    """RULE_TIMEKEY(14자리 이상)를 datetime으로 파싱한다."""
    s = str(rule_timekey).strip()
    if not s:
        return None
    s = s.ljust(14, "0")[:14]
    try:
        return datetime.strptime(s, "%Y%m%d%H%M%S")
    except ValueError:
        pass
    try:
        return datetime.strptime(s[:10], "%Y%m%d%H")
    except ValueError:
        return None


def _format_plan_time(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H")


def plan_windows_for_rule_timekey(
    rule_timekey: str,
    boundary_hour: int = DAY_BOUNDARY_HOUR,
) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """D0/D1 계획 구간을 PLAN_INFO START_TIME/END_TIME(10자리)로 반환한다.

    D0: RULE_TIMEKEY 시점 ~ 다음날 boundary_hour
    D1: D0 종료 ~ 그 다음날 boundary_hour
    """
    base = parse_rule_timekey(rule_timekey)
    if base is None:
        now = datetime.now()
        d0_start = now
        d0_end = (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).replace(
            hour=boundary_hour
        )
    else:
        d0_start = base
        day_anchor = base.replace(hour=0, minute=0, second=0, microsecond=0)
        d0_end = (day_anchor + timedelta(days=1)).replace(hour=boundary_hour)
        if d0_end <= d0_start:
            d0_end = d0_end + timedelta(days=1)
    d1_start = d0_end
    d1_end = d1_start + timedelta(days=1)
    return (
        (_format_plan_time(d0_start), _format_plan_time(d0_end)),
        (_format_plan_time(d1_start), _format_plan_time(d1_end)),
    )


def empty_snapshot_frames() -> Dict[str, pd.DataFrame]:
    return {
        "wip_info": pd.DataFrame(columns=["PLAN_PROD_KEY", "OPER_ID", "OPER_SEQ", "WIP_QTY"]),
        "uph_info": pd.DataFrame(columns=["PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD", "UPH"]),
        "eqp_qty_info": pd.DataFrame(columns=["BATCH_ID", "EQP_MODEL_CD", "TIME_SLOT", "EQP_QTY"]),
        "avail_info": pd.DataFrame(
            columns=["PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD", "AVAIL_YN"]
        ),
        "batch_tool_info": pd.DataFrame(columns=["BATCH_ID", "PLAN_PROD_KEY", "OPER_ID"]),
        "tool_qty_info": pd.DataFrame(columns=["BATCH_ID", "EQP_MODEL_CD", "TOOL_QTY"]),
        "plan_info": pd.DataFrame(
            columns=["PLAN_PROD_KEY", "OPER_ID", "START_TIME", "END_TIME", "PLAN_QTY"]
        ),
    }
