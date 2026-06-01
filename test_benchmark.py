"""벤치마크 데이터셋 평가 — test/data CSV 로더 사용 (DB 불필요)."""
from biz.services.rl_scheduler_service import RLSchedulerService
from biz.services.rl.validation.test_data_loader import TestDataLoader


class MockBenchmarkService(RLSchedulerService):
    """DB 없이 TestDataLoader로 벤치마크 입력을 공급."""

    def __init__(self, scenario: str):
        super().__init__(db_manager=None)
        self._scenario = scenario

    def fetch_data(self, rule_timekey=None):
        loader = TestDataLoader()
        return loader.load_for_env(self._scenario, rule_timekey=rule_timekey)


def run_scenario(scenario: str):
    print(f"\n{'=' * 60}\n[벤치마크] {scenario}\n{'=' * 60}")
    service = MockBenchmarkService(scenario)
    service.evaluate_on_benchmark_dataset(benchmark_dataset=scenario)


if __name__ == "__main__":
    loader = TestDataLoader()
    for scenario in loader.list_scenarios():
        run_scenario(scenario)
