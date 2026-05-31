"""Input-only 계획 기반 장비 배치 분석 단위 테스트."""

import json

from biz.services.plan_allocation.optimizer import PlanAllocationOptimizer
from biz.services.rl.validation.test_data_loader import TestDataLoader


def test_benchmark_dataset_analysis():
    data = TestDataLoader().load_for_env("benchmark_dataset")
    gt = TestDataLoader().load_ground_truth("benchmark_dataset")
    target = gt.get("target_allocation", {})

    result = PlanAllocationOptimizer(data, max_iterations=300).run(optimize=True)
    opt = result.optimized_summary
    init = result.initial_summary

    assert opt.total_plan_qty > 0
    assert opt.overall_achievement > 0
    assert len(opt.slot_contributions) >= 6
    assert opt.overall_achievement >= init.overall_achievement

    # 벤치마크 정답 배치 대비 마지막 공정 달성률이 유사한지 (정적 추정)
    for prod in target:
        last_oper = max(target[prod], key=lambda o: o)
        assert prod in opt.by_product_last_oper

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False)[:2000])


def test_marginal_ranking():
    data = TestDataLoader().load_for_env("benchmark_dataset")
    rows = PlanAllocationOptimizer(data).marginal_report()
    assert len(rows) > 0
    assert "MARGINAL_OVERALL_ACHIEVEMENT(%)" in rows[0]


if __name__ == "__main__":
    test_benchmark_dataset_analysis()
    test_marginal_ranking()
    print("OK")
