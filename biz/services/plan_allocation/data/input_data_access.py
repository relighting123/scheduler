"""Input 스냅샷 7종 DB 조회."""

import pandas as pd

from biz.services.plan_allocation.data.input_data_constants import (
    INPUT_DATA_COLUMNS,
    INPUT_DATA_TABLE,
    empty_snapshot_frames,
)
from biz.services.plan_allocation.data.input_data_snapshot_pandas import (
    transform_input_data_snapshot,
)
from biz.services.plan_allocation.data.input_data_snapshot_sql import fetch_snapshot_from_db

DEFAULT_RULE_TIMEKEY = "20251020070000"

_INPUT_DATA_SELECT = (
    "SELECT RULE_TIMEKEY, FAC_ID, BATCH_ID, PLAN_PROD_KEY, OPER_ID, "
    "OPER_SEQ, EQP_MODEL_CD, GBN_CD, ATTR_VAL "
    f"FROM {INPUT_DATA_TABLE}"
)


class InputDataAccess:
    """RTS_LINEDSDB_INF에서 Input 7종 스냅샷을 로드한다."""

    @staticmethod
    def _row_value(row, column: str):
        if not row:
            return None
        target = column.upper()
        for key, value in row.items():
            if str(key).upper() == target:
                return value
        return None

    def __init__(self, db_manager, default_rule_timekey=DEFAULT_RULE_TIMEKEY):
        self.db = db_manager
        self.default_rule_timekey = default_rule_timekey

    def resolve_rule_timekey(self, rule_timekey=None):
        if rule_timekey and str(rule_timekey) not in ("", "N/A"):
            return str(rule_timekey)
        try:
            row = self.db.select_one(
                f"SELECT MAX(RULE_TIMEKEY) AS RULE_TIMEKEY FROM {INPUT_DATA_TABLE}"
            )
            tk_val = self._row_value(row, "RULE_TIMEKEY")
            if tk_val:
                return str(tk_val)
        except Exception:
            pass
        return self.default_rule_timekey

    def fetch_input_data_rows(self, rule_timekey=None):
        if rule_timekey is not None:
            resolved = str(rule_timekey)
            rows = self.db.select_list(
                f"{_INPUT_DATA_SELECT} WHERE RULE_TIMEKEY = :tk",
                {"tk": resolved},
            )
        else:
            rows = self.db.select_list(_INPUT_DATA_SELECT)

        if not rows:
            return pd.DataFrame(columns=INPUT_DATA_COLUMNS)
        return pd.DataFrame(rows, columns=INPUT_DATA_COLUMNS)

    def fetch_data(self, rule_timekey=None):
        resolved_tk = self.resolve_rule_timekey(rule_timekey)
        try:
            data = fetch_snapshot_from_db(self.db, resolved_tk)
            print(f"[성공] DB에서 RULE_TIMEKEY={resolved_tk} 스냅샷을 조회했습니다.")
            return data
        except Exception as exc:
            print(f"[경고] DB 연동 실패: {exc}")
            try:
                df = self.fetch_input_data_rows(rule_timekey=resolved_tk)
                return transform_input_data_snapshot(df, rule_timekey=resolved_tk)
            except Exception:
                return empty_snapshot_frames()
