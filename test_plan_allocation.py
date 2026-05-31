"""Input-only 계획 기반 장비 배치 분석 단위 테스트."""

import json

from biz.services.plan_allocation.optimizer import PlanAllocationOptimizer
from biz.services.plan_allocation.test_data_loader import TestDataLoader


def test_benchmark_dataset_analysis():
    data = TestDataLoader().load_snapshot("benchmark_dataset")
    TestDataLoader().load_ground_truth("benchmark_dataset")

    result = PlanAllocationOptimizer(data, max_iterations=300).run(optimize=True)
    opt = result.optimized_summary
    init = result.initial_summary

    assert opt.total_plan_qty > 0
    assert opt.overall_achievement > 0
    assert len(opt.slot_contributions) >= 6
    assert opt.overall_achievement >= init.overall_achievement

    for prod in opt.by_product_last_oper:
        assert prod in opt.by_product_last_oper

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False)[:1500])


def test_marginal_ranking():
    data = TestDataLoader().load_snapshot("benchmark_dataset")
    rows = PlanAllocationOptimizer(data).marginal_report()
    assert len(rows) > 0
    assert "MARGINAL_OVERALL_ACHIEVEMENT(%)" in rows[0]


if __name__ == "__main__":
    test_benchmark_dataset_analysis()
    test_marginal_ranking()
    print("OK")
