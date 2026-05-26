"""Benchmark scenarios stored in DB (input tables + BENCHMARK_SCENARIO metadata)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from biz.services.rl.utils.data_access import TrainingDataAccess

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CSV_ROOT = WORKSPACE_ROOT / "test" / "data"

INPUT_TABLE_FILES = [
    "wip_info.csv",
    "uph_info.csv",
    "eqp_qty_info.csv",
    "avail_info.csv",
    "batch_tool_info.csv",
    "tool_qty_info.csv",
    "plan_info.csv",
]

DATA_KEY_BY_FILE = {
    "wip_info.csv": "wip_info",
    "uph_info.csv": "uph_info",
    "eqp_qty_info.csv": "eqp_qty_info",
    "avail_info.csv": "avail_info",
    "batch_tool_info.csv": "batch_tool_info",
    "tool_qty_info.csv": "tool_qty_info",
    "plan_info.csv": "plan_info",
}

ORACLE_TABLE_BY_KEY = {
    "wip_info": "WIP_INFO",
    "uph_info": "UPH_INFO",
    "eqp_qty_info": "EQP_QTY_INFO",
    "avail_info": "AVAIL_INFO",
    "batch_tool_info": "BATCH_TOOL_INFO",
    "tool_qty_info": "TOOL_QTY_INFO",
    "plan_info": "PLAN_INFO",
}

INSERT_SQL_BY_KEY = {
    "wip_info": """
        INSERT INTO WIP_INFO (RULE_TIMEKEY, PLAN_PROD_KEY, OPER_ID, OPER_SEQ, WIP_QTY)
        VALUES (:RULE_TIMEKEY, :PLAN_PROD_KEY, :OPER_ID, :OPER_SEQ, :WIP_QTY)
    """,
    "uph_info": """
        INSERT INTO UPH_INFO (RULE_TIMEKEY, PLAN_PROD_KEY, OPER_ID, EQP_MODEL_CD, UPH)
        VALUES (:RULE_TIMEKEY, :PLAN_PROD_KEY, :OPER_ID, :EQP_MODEL_CD, :UPH)
    """,
    "eqp_qty_info": """
        INSERT INTO EQP_QTY_INFO (RULE_TIMEKEY, BATCH_ID, EQP_MODEL_CD, TIME_SLOT, EQP_QTY)
        VALUES (:RULE_TIMEKEY, :BATCH_ID, :EQP_MODEL_CD, :TIME_SLOT, :EQP_QTY)
    """,
    "avail_info": """
        INSERT INTO AVAIL_INFO (RULE_TIMEKEY, PLAN_PROD_KEY, OPER_ID, EQP_MODEL_CD, AVAIL_YN)
        VALUES (:RULE_TIMEKEY, :PLAN_PROD_KEY, :OPER_ID, :EQP_MODEL_CD, :AVAIL_YN)
    """,
    "batch_tool_info": """
        INSERT INTO BATCH_TOOL_INFO (RULE_TIMEKEY, BATCH_ID, PLAN_PROD_KEY, OPER_ID)
        VALUES (:RULE_TIMEKEY, :BATCH_ID, :PLAN_PROD_KEY, :OPER_ID)
    """,
    "tool_qty_info": """
        INSERT INTO TOOL_QTY_INFO (RULE_TIMEKEY, BATCH_ID, EQP_MODEL_CD, TOOL_QTY)
        VALUES (:RULE_TIMEKEY, :BATCH_ID, :EQP_MODEL_CD, :TOOL_QTY)
    """,
    "plan_info": """
        INSERT INTO PLAN_INFO (
            RULE_TIMEKEY, PLAN_PROD_KEY, OPER_ID, START_TIME, END_TIME, PLAN_QTY
        )
        VALUES (
            :RULE_TIMEKEY, :PLAN_PROD_KEY, :OPER_ID, :START_TIME, :END_TIME, :PLAN_QTY
        )
    """,
}

NUMERIC_COLUMNS = {
    "wip_info": ["OPER_SEQ", "WIP_QTY"],
    "uph_info": ["UPH"],
    "eqp_qty_info": ["EQP_QTY"],
    "tool_qty_info": ["TOOL_QTY"],
    "plan_info": ["PLAN_QTY"],
}


class BenchmarkDataAccess:
    """Read/write benchmark scenarios via DB."""

    def __init__(self, db_manager, data_access: Optional[TrainingDataAccess] = None):
        if db_manager is None:
            raise ValueError("BenchmarkDataAccess requires a database connection.")
        self.db = db_manager
        self.data = data_access or TrainingDataAccess(db_manager)

    @staticmethod
    def _lob_to_str(value) -> str:
        if value is None:
            return ""
        if hasattr(value, "read"):
            return value.read()
        return str(value)

    def list_scenarios(self) -> List[str]:
        try:
            rows = self.db.select_list(
                "SELECT SCENARIO_ID FROM BENCHMARK_SCENARIO ORDER BY SCENARIO_ID"
            )
            return [str(r["scenario_id"]) for r in rows if r.get("scenario_id")]
        except Exception:
            return []

    def get_scenario_row(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        return self.db.select_one(
            """
            SELECT SCENARIO_ID, RULE_TIMEKEY, DESCRIPTION, GROUND_TRUTH_JSON
            FROM BENCHMARK_SCENARIO
            WHERE SCENARIO_ID = :scenario_id
            """,
            {"scenario_id": scenario_id},
        )

    def load_ground_truth(self, scenario_id: str = "benchmark_dataset") -> Dict[str, Any]:
        row = self.get_scenario_row(scenario_id)
        if not row:
            raise FileNotFoundError(
                f"DB에 벤치마크 시나리오가 없습니다: {scenario_id}. "
                "seed_benchmark_scenarios()를 먼저 실행하세요."
            )
        payload = json.loads(self._lob_to_str(row.get("ground_truth_json")))
        payload.setdefault("scenario_id", scenario_id)
        payload.setdefault("rule_timekey", str(row.get("rule_timekey", "")))
        return payload

    def resolve_rule_timekey(self, scenario_id: str) -> str:
        row = self.get_scenario_row(scenario_id)
        if row and row.get("rule_timekey"):
            return str(row["rule_timekey"])
        ground_truth = self.load_ground_truth(scenario_id)
        rule_timekey = ground_truth.get("rule_timekey")
        if not rule_timekey:
            raise ValueError(f"시나리오 {scenario_id}에 RULE_TIMEKEY가 없습니다.")
        return str(rule_timekey)

    def fetch_scenario_input(self, scenario_id: str) -> Dict[str, pd.DataFrame]:
        """Load env input dict for a benchmark scenario from DB (same path as train/infer)."""
        rule_timekey = self.resolve_rule_timekey(scenario_id)
        return self.data.fetch_data(rule_timekey=rule_timekey)

    def _delete_scenario_input_rows(self, rule_timekey: str) -> None:
        for table in ORACLE_TABLE_BY_KEY.values():
            self.db.execute(
                f"DELETE FROM {table} WHERE RULE_TIMEKEY = :tk",
                {"tk": str(rule_timekey)},
            )

    @staticmethod
    def _coerce_numeric_columns(table_key: str, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        for col in NUMERIC_COLUMNS.get(table_key, []):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    def _insert_table_rows(self, table_key: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        params = []
        for _, row in df.iterrows():
            record = {str(k).upper(): row[k] for k in df.columns}
            for col in NUMERIC_COLUMNS.get(table_key, []):
                upper = col.upper()
                if upper in record and pd.notna(record[upper]):
                    record[upper] = float(record[upper])
            params.append(record)
        self.db.bulk_execute(INSERT_SQL_BY_KEY[table_key], params)
        return len(params)

    def upsert_scenario_metadata(
        self,
        scenario_id: str,
        rule_timekey: str,
        ground_truth: Dict[str, Any],
        description: Optional[str] = None,
    ) -> None:
        desc = description or ground_truth.get("description", "")
        payload = json.dumps(ground_truth, ensure_ascii=False)
        existing = self.get_scenario_row(scenario_id)
        if existing:
            self.db.execute(
                """
                UPDATE BENCHMARK_SCENARIO
                SET RULE_TIMEKEY = :rule_timekey,
                    DESCRIPTION = :description,
                    GROUND_TRUTH_JSON = :ground_truth_json
                WHERE SCENARIO_ID = :scenario_id
                """,
                {
                    "scenario_id": scenario_id,
                    "rule_timekey": str(rule_timekey),
                    "description": str(desc)[:500],
                    "ground_truth_json": payload,
                },
            )
        else:
            self.db.execute(
                """
                INSERT INTO BENCHMARK_SCENARIO (
                    SCENARIO_ID, RULE_TIMEKEY, DESCRIPTION, GROUND_TRUTH_JSON
                ) VALUES (
                    :scenario_id, :rule_timekey, :description, :ground_truth_json
                )
                """,
                {
                    "scenario_id": scenario_id,
                    "rule_timekey": str(rule_timekey),
                    "description": str(desc)[:500],
                    "ground_truth_json": payload,
                },
            )

    def seed_scenario_from_directory(
        self,
        scenario_id: str,
        csv_root: Optional[Path] = None,
        reload: bool = True,
    ) -> str:
        """Load one scenario's CSV files from disk into the input tables."""
        root = Path(csv_root) if csv_root else DEFAULT_CSV_ROOT
        scenario_dir = root / scenario_id
        if not scenario_dir.is_dir():
            raise FileNotFoundError(f"테스트 시나리오 폴더가 없습니다: {scenario_dir}")

        gt_path = scenario_dir / "ground_truth.json"
        if not gt_path.is_file():
            raise FileNotFoundError(f"정답 파일이 없습니다: {gt_path}")
        with open(gt_path, encoding="utf-8") as handle:
            ground_truth = json.load(handle)

        rule_timekey = str(ground_truth.get("rule_timekey", "")).strip()
        if not rule_timekey:
            raise ValueError(f"{scenario_id}: ground_truth.json에 rule_timekey가 필요합니다.")

        if reload:
            self._delete_scenario_input_rows(rule_timekey)

        inserted = 0
        for filename in INPUT_TABLE_FILES:
            path = scenario_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"입력 파일이 없습니다: {path}")
            table_key = DATA_KEY_BY_FILE[filename]
            df = pd.read_csv(path, dtype=str)
            df = self._coerce_numeric_columns(table_key, df)
            if "RULE_TIMEKEY" in df.columns:
                df = df[df["RULE_TIMEKEY"] == rule_timekey].copy()
            inserted += self._insert_table_rows(table_key, df)

        self.upsert_scenario_metadata(
            scenario_id=scenario_id,
            rule_timekey=rule_timekey,
            ground_truth=ground_truth,
        )
        print(
            f"[벤치마크 DB 적재] {scenario_id} "
            f"(RULE_TIMEKEY={rule_timekey}, rows={inserted})"
        )
        return rule_timekey

    def seed_all_from_directory(
        self,
        csv_root: Optional[Path] = None,
        reload: bool = True,
        scenarios: Optional[List[str]] = None,
    ) -> List[str]:
        """Seed every scenario folder that contains ground_truth.json."""
        root = Path(csv_root) if csv_root else DEFAULT_CSV_ROOT
        if scenarios is None:
            if not root.is_dir():
                return []
            scenarios = sorted(
                d.name
                for d in root.iterdir()
                if d.is_dir() and (d / "ground_truth.json").exists()
            )

        loaded = []
        for scenario_id in scenarios:
            self.seed_scenario_from_directory(
                scenario_id,
                csv_root=root,
                reload=reload,
            )
            loaded.append(scenario_id)
        print(f"[벤치마크 DB 적재 완료] {len(loaded)}개 시나리오: {', '.join(loaded)}")
        return loaded

    def ensure_scenario_loaded(
        self,
        scenario_id: str,
        csv_root: Optional[Path] = None,
    ) -> None:
        """Load scenario from CSV into DB only when metadata row is missing."""
        if self.get_scenario_row(scenario_id):
            return
        self.seed_scenario_from_directory(scenario_id, csv_root=csv_root, reload=True)

    def ensure_all_loaded(
        self,
        csv_root: Optional[Path] = None,
        scenarios: Optional[List[str]] = None,
    ) -> List[str]:
        root = Path(csv_root) if csv_root else DEFAULT_CSV_ROOT
        if scenarios is None:
            if not root.is_dir():
                return self.list_scenarios()
            scenarios = sorted(
                d.name
                for d in root.iterdir()
                if d.is_dir() and (d / "ground_truth.json").exists()
            )
        for scenario_id in scenarios:
            self.ensure_scenario_loaded(scenario_id, csv_root=root)
        return self.list_scenarios()
