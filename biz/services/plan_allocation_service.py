"""계획·Input 기반 장비 배치 분석/최적화 서비스."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from biz.services.plan_allocation.data import InputDataAccess
from biz.services.plan_allocation.optimizer import PlanAllocationOptimizer
from biz.services.plan_allocation.report import format_allocation_report, result_as_json
from biz.services.plan_allocation.test_data_loader import TestDataLoader

logger = logging.getLogger(__name__)


def run_plan_allocation(
    repo=None,
    rule_timekey: Optional[str] = None,
    *,
    scenario: Optional[str] = None,
    optimize: bool = True,
    include_marginal: bool = True,
    max_iterations: int = 200,
) -> Dict[str, Any]:
    """Input 7종만으로 정적 계획 달성·장비 배치 분석을 수행한다."""
    data = _load_input_data(repo, rule_timekey=rule_timekey, scenario=scenario)
    optimizer = PlanAllocationOptimizer(data, max_iterations=max_iterations)
    result = optimizer.run(optimize=optimize)
    marginal = optimizer.marginal_report() if include_marginal else None

    report_text = format_allocation_report(result)
    print(report_text)
    logger.info(report_text)

    return result_as_json(result, marginal=marginal)


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
