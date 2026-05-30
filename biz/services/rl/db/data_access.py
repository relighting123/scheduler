"""학습 및 추론용 DB 스냅샷 조회와 RULE_TIMEKEY 필터링."""

import pandas as pd

from biz.services.rl.db.linedb_queries import fetch_snapshot_from_db
from biz.services.rl.db.linedb_transform import (
    LINEDB_COLUMNS,
    LINEDB_TABLE,
    empty_snapshot_frames,
    transform_linedb_snapshot,
)

DEFAULT_RULE_TIMEKEY = "20251020070000"

_LINEDB_SELECT = (
    "SELECT RULE_TIMEKEY, FAC_ID, BATCH_ID, PLAN_PROD_KEY, OPER_ID, "
    "OPER_SEQ, EQP_MODEL_CD, GBN_CD, ATTR_VAL "
    f"FROM {LINEDB_TABLE}"
)


class TrainingDataAccess:
    """DB에서 스케줄러 학습 스냅샷을 로드하고 필터링한다."""

    @staticmethod
    def _row_value(row, column: str):
        """Oracle rowfactory 소문자 컬럼명 호환."""
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
                f"SELECT MAX(RULE_TIMEKEY) AS RULE_TIMEKEY FROM {LINEDB_TABLE}"
            )
            tk_val = self._row_value(row, "RULE_TIMEKEY")
            if tk_val:
                return str(tk_val)
        except Exception:
            pass
        return self.default_rule_timekey

    @staticmethod
    def normalize_timekey_arg(value):
        if value is None:
            return None
        s = str(value).strip()
        if s in ("", "N/A"):
            return None
        return s

    def list_rule_timekeys_in_range(self, from_rule_timekey=None, to_rule_timekey=None):
        from_tk = self.normalize_timekey_arg(from_rule_timekey)
        to_tk = self.normalize_timekey_arg(to_rule_timekey)

        if not from_tk and not to_tk:
            return [self.resolve_rule_timekey(None)]
        if from_tk and not to_tk:
            to_tk = from_tk
        elif to_tk and not from_tk:
            from_tk = to_tk

        try:
            rows = self.db.select_list(
                f"SELECT DISTINCT RULE_TIMEKEY FROM {LINEDB_TABLE} ORDER BY RULE_TIMEKEY"
            )
            keys = []
            for r in rows:
                tk_val = self._row_value(r, "RULE_TIMEKEY")
                if tk_val and from_tk <= str(tk_val) <= to_tk:
                    keys.append(str(tk_val))
            if keys:
                return keys
        except Exception:
            pass
        return [self.resolve_rule_timekey(to_tk)]

    def fetch_linedb_rows(self, rule_timekey=None):
        """RTS_LINEDSDB_INF 원본 EAV 행 조회 (디버그·pandas 변환용)."""
        if rule_timekey is not None:
            resolved = str(rule_timekey)
            rows = self.db.select_list(
                f"{_LINEDB_SELECT} WHERE RULE_TIMEKEY = :tk",
                {"tk": resolved},
            )
        else:
            rows = self.db.select_list(_LINEDB_SELECT)

        if not rows:
            return pd.DataFrame(columns=LINEDB_COLUMNS)
        return pd.DataFrame(rows, columns=LINEDB_COLUMNS)

    def fetch_snapshot(self, rule_timekey: str):
        """SQL 집계·필터로 env 입력 7종 DataFrame을 조회한다."""
        return fetch_snapshot_from_db(self.db, rule_timekey)

    def fetch_raw_tables(self):
        """하위 호환: RULE_TIMEKEY 포함 7종 테이블 형태 raw 프레임."""
        try:
            rows = self.db.select_list(
                f"SELECT DISTINCT RULE_TIMEKEY FROM {LINEDB_TABLE} ORDER BY RULE_TIMEKEY"
            )
            keys = [
                str(self._row_value(r, "RULE_TIMEKEY"))
                for r in rows
                if self._row_value(r, "RULE_TIMEKEY")
            ]
        except Exception:
            keys = []

        if not keys:
            return self._empty_raw_frames()

        raw = {}
        for tk in keys:
            snap = self.fetch_snapshot(tk)
            for key, part in snap.items():
                part = part.copy()
                part.insert(0, "RULE_TIMEKEY", tk)
                if key not in raw:
                    raw[key] = part
                else:
                    raw[key] = pd.concat([raw[key], part], ignore_index=True)
        empty = self._empty_raw_frames()
        return {k: raw.get(k, empty[k]) for k in empty}

    @staticmethod
    def filter_data_by_rule_timekey(data, rule_timekey):
        filtered = {}
        for key, df in data.items():
            if df is None or df.empty:
                filtered[key] = df
                continue
            if "RULE_TIMEKEY" not in df.columns:
                filtered[key] = df
                continue
            snapshot = df[df["RULE_TIMEKEY"] == rule_timekey].copy()
            filtered[key] = snapshot.drop(columns=["RULE_TIMEKEY"], errors="ignore")
        return filtered

    def _empty_raw_frames(self):
        frames = {}
        for key, df in empty_snapshot_frames().items():
            frames[key] = pd.DataFrame(columns=["RULE_TIMEKEY"] + list(df.columns))
        return frames

    def fetch_data(self, rule_timekey=None):
        resolved_tk = self.resolve_rule_timekey(rule_timekey)
        try:
            data = self.fetch_snapshot(resolved_tk)
            print(f"[성공] DB에서 RULE_TIMEKEY={resolved_tk} 스냅샷 데이터를 조회했습니다.")
            return data
        except Exception as exc:
            print(f"[경고] DB 연동 실패 (또는 테이블 없음): {exc}")
            print(
                "데이터를 조회할 수 없습니다. DB 초기화(init_db_scenario)가 올바르게 수행되었는지 확인하세요."
            )
            try:
                df = self.fetch_linedb_rows(rule_timekey=resolved_tk)
                return transform_linedb_snapshot(df, rule_timekey=resolved_tk)
            except Exception:
                return empty_snapshot_frames()

    def fetch_training_snapshots(
        self,
        from_rule_timekey=None,
        to_rule_timekey=None,
        rule_timekey=None,
    ):
        single = self.normalize_timekey_arg(rule_timekey)
        if single:
            return [self.fetch_data(rule_timekey=single)]
        keys = self.list_rule_timekeys_in_range(from_rule_timekey, to_rule_timekey)
        snapshots = [self.fetch_data(rule_timekey=k) for k in keys]
        print(
            f"[학습 데이터] RULE_TIMEKEY {len(snapshots)}개 스냅샷 "
            f"({', '.join(keys)})"
        )
        return snapshots
