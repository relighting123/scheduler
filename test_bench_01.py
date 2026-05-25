"""벤치마크 bench_01 평가 — test/data/bench_01 (DB 불필요)."""
from biz.services.rl.benchmark_scenarios import DEFAULT_BENCHMARK_SCENARIO
from biz.services.rl_scheduler_service import RLSchedulerService
from biz.services.rl.test_data_loader import TestDataLoader


class MockBenchmarkDatasetService(RLSchedulerService):
    def init_benchmark_dataset_scenario(self):
        print(f"\n[Mock] 벤치마크 시나리오 (test/data/{DEFAULT_BENCHMARK_SCENARIO})")

    def fetch_data(self, rule_timekey=None):
        loader = TestDataLoader()
        return loader.load_for_env(DEFAULT_BENCHMARK_SCENARIO, rule_timekey=rule_timekey)


if __name__ == "__main__":
    print("[1] bench_01 평가 서비스 초기화...")
    service = MockBenchmarkDatasetService(db_manager=None)

    print("\n[2] bench_01 평가 (정답 vs 휴리스틱 vs RL)...")
    service.evaluate_on_benchmark_dataset(benchmark_dataset="bench_01")
    print("\n[완료]")
