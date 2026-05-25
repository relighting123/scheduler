"""테스트 폴더(test/data)에서 벤치마크 입력·정답 데이터를 체계적으로 로드합니다."""
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from biz.services.rl.benchmark_scenarios import (
    DEFAULT_BENCHMARK_SCENARIO,
    resolve_scenario_id,
)

TEST_DATA_ROOT = Path(__file__).resolve().parents[3] / "test" / "data"

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
    """test/data/<scenario>/ 하위 CSV·ground_truth.json 관리."""

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
        resolved = resolve_scenario_id(scenario)
        return self.root / resolved

    def load_ground_truth(
        self, scenario: str = DEFAULT_BENCHMARK_SCENARIO
    ) -> Dict[str, Any]:
        path = self.scenario_dir(scenario) / "ground_truth.json"
        if not path.is_file():
            raise FileNotFoundError(f"정답 파일이 없습니다: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def load_input_tables(
        self,
        scenario: str = DEFAULT_BENCHMARK_SCENARIO,
        rule_timekey: Optional[str] = None,
        drop_timekey: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """시나리오 입력 CSV 7종을 로드하고 RULE_TIMEKEY로 필터링합니다."""
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
        scenario: str = DEFAULT_BENCHMARK_SCENARIO,
        rule_timekey: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """SchedulerEnv에 바로 넣을 수 있는 형태(RULE_TIMEKEY 제거)로 반환."""
        return self.load_input_tables(
            scenario=scenario,
            rule_timekey=rule_timekey,
            drop_timekey=True,
        )
