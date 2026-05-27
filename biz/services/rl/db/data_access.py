"""학습 및 추론용 DB 스냅샷 조회와 RULE_TIMEKEY 필터링."""

import pandas as pd

DEFAULT_RULE_TIMEKEY = "20251020070000"


class TrainingDataAccess:
    """DB에서 스케줄러 학습 스냅샷을 로드하고 필터링한다."""

    def __init__(self, db_manager, default_rule_timekey=DEFAULT_RULE_TIMEKEY):
        self.db = db_manager
        self.default_rule_timekey = default_rule_timekey

    def resolve_rule_timekey(self, rule_timekey=None):
        if rule_timekey and str(rule_timekey) not in ("", "N/A"):
            return str(rule_timekey)
        try:
            row = self.db.select_one(
                "SELECT MAX(RULE_TIMEKEY) AS RULE_TIMEKEY FROM WIP_INFO"
            )
            if row and row.get("RULE_TIMEKEY"):
                return str(row["RULE_TIMEKEY"])
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
                "SELECT DISTINCT RULE_TIMEKEY FROM WIP_INFO ORDER BY RULE_TIMEKEY"
            )
            keys = [
                str(r["RULE_TIMEKEY"])
                for r in rows
                if r.get("RULE_TIMEKEY") and from_tk <= str(r["RULE_TIMEKEY"]) <= to_tk
            ]
            if keys:
                return keys
        except Exception:
            pass
        return [self.resolve_rule_timekey(to_tk)]

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
        return {
            "wip_info": pd.DataFrame(
                columns=["RULE_TIMEKEY", "PLAN_PROD_KEY", "OPER_ID", "OPER_SEQ", "WIP_QTY"]
            ),
            "uph_info": pd.DataFrame(
                columns=["RULE_TIMEKEY", "PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD", "UPH"]
            ),
            "eqp_qty_info": pd.DataFrame(
                columns=["RULE_TIMEKEY", "BATCH_ID", "EQP_MODEL_CD", "TIME_SLOT", "EQP_QTY"]
            ),
            "avail_info": pd.DataFrame(
                columns=["RULE_TIMEKEY", "PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD", "AVAIL_YN"]
            ),
            "batch_tool_info": pd.DataFrame(
                columns=["RULE_TIMEKEY", "BATCH_ID", "PLAN_PROD_KEY", "OPER_ID"]
            ),
            "tool_qty_info": pd.DataFrame(
                columns=["RULE_TIMEKEY", "BATCH_ID", "EQP_MODEL_CD", "TOOL_QTY"]
            ),
            "plan_info": pd.DataFrame(
                columns=[
                    "RULE_TIMEKEY",
                    "PLAN_PROD_KEY",
                    "OPER_ID",
                    "START_TIME",
                    "END_TIME",
                    "PLAN_QTY",
                ]
            ),
        }

    def fetch_raw_tables(self):
        wip_data = pd.DataFrame(
            self.db.select_list(
                "SELECT RULE_TIMEKEY, PLAN_PROD_KEY, OPER_ID, OPER_SEQ, WIP_QTY FROM WIP_INFO"
            ),
            columns=["RULE_TIMEKEY", "PLAN_PROD_KEY", "OPER_ID", "OPER_SEQ", "WIP_QTY"],
        )
        uph_data = pd.DataFrame(
            self.db.select_list(
                "SELECT RULE_TIMEKEY, PLAN_PROD_KEY, OPER_ID, EQP_MODEL_CD, UPH FROM UPH_INFO"
            ),
            columns=["RULE_TIMEKEY", "PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD", "UPH"],
        )
        eqp_qty_data = pd.DataFrame(
            self.db.select_list(
                "SELECT RULE_TIMEKEY, BATCH_ID, EQP_MODEL_CD, TIME_SLOT, EQP_QTY FROM EQP_QTY_INFO"
            ),
            columns=["RULE_TIMEKEY", "BATCH_ID", "EQP_MODEL_CD", "TIME_SLOT", "EQP_QTY"],
        )
        avail_data = pd.DataFrame(
            self.db.select_list(
                "SELECT RULE_TIMEKEY, PLAN_PROD_KEY, OPER_ID, EQP_MODEL_CD, AVAIL_YN FROM AVAIL_INFO"
            ),
            columns=["RULE_TIMEKEY", "PLAN_PROD_KEY", "OPER_ID", "EQP_MODEL_CD", "AVAIL_YN"],
        )
        batch_tool_data = pd.DataFrame(
            self.db.select_list(
                "SELECT RULE_TIMEKEY, BATCH_ID, PLAN_PROD_KEY, OPER_ID FROM BATCH_TOOL_INFO"
            ),
            columns=["RULE_TIMEKEY", "BATCH_ID", "PLAN_PROD_KEY", "OPER_ID"],
        )
        tool_qty_data = pd.DataFrame(
            self.db.select_list(
                "SELECT RULE_TIMEKEY, BATCH_ID, EQP_MODEL_CD, TOOL_QTY FROM TOOL_QTY_INFO"
            ),
            columns=["RULE_TIMEKEY", "BATCH_ID", "EQP_MODEL_CD", "TOOL_QTY"],
        )
        plan_data = pd.DataFrame(
            self.db.select_list(
                "SELECT RULE_TIMEKEY, PLAN_PROD_KEY, OPER_ID, START_TIME, END_TIME, PLAN_QTY FROM PLAN_INFO"
            ),
            columns=[
                "RULE_TIMEKEY",
                "PLAN_PROD_KEY",
                "OPER_ID",
                "START_TIME",
                "END_TIME",
                "PLAN_QTY",
            ],
        )

        empty = self._empty_raw_frames()
        return {
            "wip_info": wip_data if not wip_data.empty else empty["wip_info"],
            "uph_info": uph_data if not uph_data.empty else empty["uph_info"],
            "eqp_qty_info": eqp_qty_data if not eqp_qty_data.empty else empty["eqp_qty_info"],
            "avail_info": avail_data if not avail_data.empty else empty["avail_info"],
            "batch_tool_info": batch_tool_data
            if not batch_tool_data.empty
            else empty["batch_tool_info"],
            "tool_qty_info": tool_qty_data if not tool_qty_data.empty else empty["tool_qty_info"],
            "plan_info": plan_data if not plan_data.empty else empty["plan_info"],
        }

    def fetch_data(self, rule_timekey=None):
        resolved_tk = self.resolve_rule_timekey(rule_timekey)
        try:
            raw = self.fetch_raw_tables()
            data = self.filter_data_by_rule_timekey(raw, resolved_tk)
            print(f"[성공] DB에서 RULE_TIMEKEY={resolved_tk} 스냅샷 데이터를 조회했습니다.")
            return data
        except Exception as exc:
            print(f"[경고] DB 연동 실패 (또는 테이블 없음): {exc}")
            print(
                "데이터를 조회할 수 없습니다. DB 초기화(init_db_scenario)가 올바르게 수행되었는지 확인하세요."
            )
            return self.filter_data_by_rule_timekey(self._empty_raw_frames(), resolved_tk)

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
