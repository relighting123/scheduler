"""벤치마크용 CSV 입력·ground_truth 로드 (`test/data/<scenario>/`)."""
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
TEST_DATA_ROOT = WORKSPACE_ROOT / "test" / "data"

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


class TestDataLoader:
    """Manage CSV scenarios under test/data/<scenario>/."""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else TEST_DATA_ROOT

    def list_scenarios(self) -> list:
        if not self.root.is_dir():
            return []
        return sorted(
            d.name
            for d in self.root.iterdir()
            if d.is_dir() and (d / "ground_truth.json").exists()
        )

    def scenario_dir(self, scenario: str) -> Path:
        return self.root / scenario

    def load_ground_truth(self, scenario: str = "benchmark_dataset") -> Dict[str, Any]:
        path = self.scenario_dir(scenario) / "ground_truth.json"
        if not path.is_file():
            raise FileNotFoundError(f"정답 파일이 없습니다: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def load_input_tables(
        self,
        scenario: str = "benchmark_dataset",
        rule_timekey: Optional[str] = None,
        drop_timekey: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        base = self.scenario_dir(scenario)
        if not base.is_dir():
            raise FileNotFoundError(f"테스트 시나리오 폴더가 없습니다: {base}")

        gt = self.load_ground_truth(scenario)
        resolved_tk = rule_timekey or gt.get("rule_timekey")
        if not resolved_tk:
            raise ValueError(
                "RULE_TIMEKEY를 지정하거나 ground_truth.json에 rule_timekey가 필요합니다."
            )

        result: Dict[str, pd.DataFrame] = {}
        for filename in INPUT_TABLE_FILES:
            path = base / filename
            if not path.is_file():
                raise FileNotFoundError(f"입력 파일이 없습니다: {path}")
            df = pd.read_csv(path, dtype=str)
            key = DATA_KEY_BY_FILE[filename]
            if "RULE_TIMEKEY" in df.columns:
                df = df[df["RULE_TIMEKEY"] == str(resolved_tk)].copy()
                if drop_timekey:
                    df = df.drop(columns=["RULE_TIMEKEY"], errors="ignore")
            result[key] = self._coerce_numeric_columns(key, df)

        return result

    @staticmethod
    def _coerce_numeric_columns(table_key: str, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        numeric_cols = {
            "wip_info": ["OPER_SEQ", "WIP_QTY"],
            "uph_info": ["UPH"],
            "eqp_qty_info": ["EQP_QTY"],
            "tool_qty_info": ["TOOL_QTY"],
            "plan_info": ["PLAN_QTY"],
        }.get(table_key, [])
        for col in numeric_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    def load_for_env(
        self,
        scenario: str = "benchmark_dataset",
        rule_timekey: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        return self.load_input_tables(
            scenario=scenario,
            rule_timekey=rule_timekey,
            drop_timekey=True,
        )
