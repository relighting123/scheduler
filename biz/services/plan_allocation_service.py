"""정적 장비 배치 — Input만으로 계획제품×공정×모델별 대수 산출."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from biz.services.plan_allocation.data import InputDataAccess
from biz.services.plan_allocation.optimizer import PlanAllocationOptimizer
from biz.services.plan_allocation.report import format_allocation_report, result_as_json
from biz.services.plan_allocation.static_result import (
    format_static_allocation_table,
    save_static_allocation_files,
    static_allocation_payload,
)
from biz.services.plan_allocation.test_data_loader import TestDataLoader

logger = logging.getLogger(__name__)


def run_static_allocation(
    repo=None,
    rule_timekey: Optional[str] = None,
    *,
    scenario: Optional[str] = None,
    optimize: bool = True,
    include_marginal: bool = False,
    max_iterations: int = 200,
    output_dir: str = "output",
    verbose: bool = False,
) -> Dict[str, Any]:
    """정적으로 몇 대 배치할지 결과를 만든다 (시간대별 판단 없음).

    Returns:
        static_allocation payload (allocation_table, achievement, saved paths)
    """
    data = _load_input_data(repo, rule_timekey=rule_timekey, scenario=scenario)
    resolved_tk = rule_timekey
    if not resolved_tk and scenario:
        try:
            resolved_tk = TestDataLoader().load_ground_truth(scenario).get("rule_timekey")
        except Exception:
            pass
    if not resolved_tk and repo is not None:
        resolved_tk = InputDataAccess(repo).resolve_rule_timekey(rule_timekey)

    optimizer = PlanAllocationOptimizer(data, max_iterations=max_iterations)
    result = optimizer.run(optimize=optimize)

    rows = static_allocation_payload(result, rule_timekey=resolved_tk)
    table_text = format_static_allocation_table(rows["allocation_table"])
    print(table_text)

    paths = save_static_allocation_files(rows, output_dir=output_dir)
    rows["saved_files"] = paths
    print(f"\n저장: {paths['json']}\n      {paths['csv']}")

    if verbose:
        detail = result_as_json(
            result,
            marginal=optimizer.marginal_report() if include_marginal else None,
        )
        rows["detail"] = detail
        print(format_allocation_report(result))
    elif include_marginal:
        rows["marginal_contribution"] = optimizer.marginal_report()

    logger.info("Static allocation completed: %s rows", len(rows["allocation_table"]))
    return rows


# 하위 호환 alias
run_plan_allocation = run_static_allocation


def _load_input_data(
    repo,
    rule_timekey: Optional[str],
    scenario: Optional[str],
) -> Dict:
    if scenario:
        return TestDataLoader().load_snapshot(
            scenario=scenario, rule_timekey=rule_timekey
        )

    if repo is not None:
        return InputDataAccess(repo).fetch_data(rule_timekey=rule_timekey)

    return TestDataLoader().load_snapshot(
        scenario="benchmark_dataset", rule_timekey=rule_timekey
    )
