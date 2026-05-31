"""RTS_LINEDSDB_INF → env 스냅샷 조회 SQL (Oracle).

필터·집계·수치 변환·중복 제거는 DB에서 수행하고,
D0/D1 PLAN 구간 시각은 plan_windows_for_rule_timekey() 결과를 바인드한다.
"""

from typing import Dict, List

import pandas as pd

from biz.services.rl.db.input_data_constants import (
    GBN_ASSIGN_EQUIP,
    GBN_D0_TARGET,
    GBN_D1_TARGET,
    GBN_TOOL,
    GBN_UPH,
    GBN_WIP,
    INPUT_DATA_TABLE,
    empty_snapshot_frames,
    plan_windows_for_rule_timekey,
)

_WHERE_TK = "RULE_TIMEKEY = :tk"
_BASE = f"FROM {INPUT_DATA_TABLE} WHERE {_WHERE_TK}"

SQL_WIP_INFO = f"""
    SELECT PLAN_PROD_KEY, OPER_ID,
           MAX(OPER_SEQ) AS OPER_SEQ,
           MAX(TO_NUMBER(TRIM(ATTR_VAL))) AS WIP_QTY
    {_BASE}
      AND UPPER(TRIM(GBN_CD)) = '{GBN_WIP}'
    GROUP BY PLAN_PROD_KEY, OPER_ID
"""

SQL_UPH_INFO = f"""
    SELECT DISTINCT PLAN_PROD_KEY, OPER_ID, EQP_MODEL_CD,
           TO_NUMBER(TRIM(ATTR_VAL)) AS UPH
    {_BASE}
      AND UPPER(TRIM(GBN_CD)) = '{GBN_UPH}'
      AND TRIM(ATTR_VAL) IS NOT NULL
      AND REGEXP_LIKE(TRIM(ATTR_VAL), '^-?\\d+(\\.\\d+)?$')
      AND TO_NUMBER(TRIM(ATTR_VAL)) > 0
"""

SQL_EQP_QTY_INFO = f"""
    SELECT BATCH_ID, EQP_MODEL_CD,
           SUBSTR(RPAD(:tk, 14, '0'), 1, 10) AS TIME_SLOT,
           TO_NUMBER(TRIM(ATTR_VAL)) AS EQP_QTY
    {_BASE}
      AND UPPER(TRIM(GBN_CD)) = '{GBN_ASSIGN_EQUIP}'
      AND TRIM(ATTR_VAL) IS NOT NULL
      AND REGEXP_LIKE(TRIM(ATTR_VAL), '^-?\\d+(\\.\\d+)?$')
"""

SQL_TOOL_QTY_INFO = f"""
    SELECT BATCH_ID, EQP_MODEL_CD,
           TO_NUMBER(TRIM(ATTR_VAL)) AS TOOL_QTY
    {_BASE}
      AND UPPER(TRIM(GBN_CD)) = '{GBN_TOOL}'
      AND TRIM(ATTR_VAL) IS NOT NULL
      AND REGEXP_LIKE(TRIM(ATTR_VAL), '^-?\\d+(\\.\\d+)?$')
"""

SQL_BATCH_TOOL_INFO = f"""
    SELECT DISTINCT BATCH_ID, PLAN_PROD_KEY, OPER_ID
    {_BASE}
      AND BATCH_ID IS NOT NULL
      AND TRIM(BATCH_ID) IS NOT NULL
      AND TRIM(BATCH_ID) <> '-'
"""

SQL_AVAIL_INFO = f"""
    SELECT PLAN_PROD_KEY, OPER_ID, EQP_MODEL_CD, 'Y' AS AVAIL_YN
    {_BASE}
      AND UPPER(TRIM(GBN_CD)) = '{GBN_UPH}'
      AND TRIM(ATTR_VAL) IS NOT NULL
      AND REGEXP_LIKE(TRIM(ATTR_VAL), '^-?\\d+(\\.\\d+)?$')
      AND TO_NUMBER(TRIM(ATTR_VAL)) > 0
    UNION ALL
    SELECT DISTINCT d.PLAN_PROD_KEY, d.OPER_ID, d.EQP_MODEL_CD, 'N' AS AVAIL_YN
      FROM {INPUT_DATA_TABLE} d
     WHERE d.RULE_TIMEKEY = :tk
       AND TRIM(d.PLAN_PROD_KEY) IS NOT NULL
       AND TRIM(d.PLAN_PROD_KEY) <> '-'
       AND TRIM(d.OPER_ID) IS NOT NULL
       AND TRIM(d.OPER_ID) <> '-'
       AND TRIM(d.EQP_MODEL_CD) IS NOT NULL
       AND TRIM(d.EQP_MODEL_CD) NOT IN ('', '-')
       AND NOT EXISTS (
           SELECT 1
             FROM {INPUT_DATA_TABLE} u
            WHERE u.RULE_TIMEKEY = d.RULE_TIMEKEY
              AND UPPER(TRIM(u.GBN_CD)) = '{GBN_UPH}'
              AND TRIM(u.ATTR_VAL) IS NOT NULL
              AND REGEXP_LIKE(TRIM(u.ATTR_VAL), '^-?\\d+(\\.\\d+)?$')
              AND TO_NUMBER(TRIM(u.ATTR_VAL)) > 0
       )
"""

SQL_PLAN_INFO = f"""
    SELECT PLAN_PROD_KEY, OPER_ID,
           :d0_start AS START_TIME,
           :d0_end AS END_TIME,
           TO_NUMBER(TRIM(ATTR_VAL)) AS PLAN_QTY
    {_BASE}
      AND UPPER(TRIM(GBN_CD)) = '{GBN_D0_TARGET}'
      AND TRIM(ATTR_VAL) IS NOT NULL
      AND REGEXP_LIKE(TRIM(ATTR_VAL), '^-?\\d+(\\.\\d+)?$')
    UNION ALL
    SELECT PLAN_PROD_KEY, OPER_ID,
           :d1_start AS START_TIME,
           :d1_end AS END_TIME,
           TO_NUMBER(TRIM(ATTR_VAL)) AS PLAN_QTY
    {_BASE}
      AND UPPER(TRIM(GBN_CD)) = '{GBN_D1_TARGET}'
      AND TRIM(ATTR_VAL) IS NOT NULL
      AND REGEXP_LIKE(TRIM(ATTR_VAL), '^-?\\d+(\\.\\d+)?$')
"""

_SNAPSHOT_QUERIES = {
    "wip_info": (SQL_WIP_INFO, ["PLAN_PROD_KEY", "OPER_ID", "OPER_SEQ", "WIP_QTY"]),
    "uph_info": (SQL_UPH_INFO, ["PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD", "UPH"]),
    "eqp_qty_info": (SQL_EQP_QTY_INFO, ["BATCH_ID", "EQP_MODEL_CD", "TIME_SLOT", "EQP_QTY"]),
    "tool_qty_info": (SQL_TOOL_QTY_INFO, ["BATCH_ID", "EQP_MODEL_CD", "TOOL_QTY"]),
    "batch_tool_info": (SQL_BATCH_TOOL_INFO, ["BATCH_ID", "PLAN_PROD_KEY", "OPER_ID"]),
    "avail_info": (
        SQL_AVAIL_INFO,
        ["PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD", "AVAIL_YN"],
    ),
    "plan_info": (
        SQL_PLAN_INFO,
        ["PLAN_PROD_KEY", "OPER_ID", "START_TIME", "END_TIME", "PLAN_QTY"],
    ),
}


def snapshot_query_params(rule_timekey: str) -> Dict[str, str]:
    """스냅샷 조회 SQL 바인드 파라미터."""
    tk = str(rule_timekey).strip()
    (d0_start, d0_end), (d1_start, d1_end) = plan_windows_for_rule_timekey(tk)
    return {
        "tk": tk,
        "d0_start": d0_start,
        "d0_end": d0_end,
        "d1_start": d1_start,
        "d1_end": d1_end,
    }


def _rows_to_dataframe(rows: List[Dict], columns: List[str]) -> pd.DataFrame:
    empty = pd.DataFrame(columns=columns)
    if not rows:
        return empty
    df = pd.DataFrame(rows)
    df.columns = [str(c).upper() for c in df.columns]
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[columns]


def fetch_snapshot_from_db(db, rule_timekey: str) -> Dict[str, pd.DataFrame]:
    """RULE_TIMEKEY 스냅샷을 SQL로 조회해 env 입력 dict를 반환한다."""
    params = snapshot_query_params(rule_timekey)
    result: Dict[str, pd.DataFrame] = {}
    for key, (sql, columns) in _SNAPSHOT_QUERIES.items():
        rows = db.select_list(sql, params)
        frame = _rows_to_dataframe(rows, columns)
        result[key] = frame if not frame.empty else empty_snapshot_frames()[key]
    return result
