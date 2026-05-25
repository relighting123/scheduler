"""벤치마크 bench_02 — test/data/bench_02 (DB 불필요)."""
from biz.services.rl_scheduler_service import RLSchedulerService
from biz.services.rl.test_data_loader import TestDataLoader

SCENARIO = "bench_02"


class MockBenchmarkService(RLSchedulerService):
    def init_benchmark_dataset_scenario(self):
        print(f"\n[Mock] 벤치마크 시나리오 (test/data/{SCENARIO})")

    def fetch_data(self, rule_timekey=None):
        loader = TestDataLoader()
        return loader.load_for_env(SCENARIO, rule_timekey=rule_timekey)


if __name__ == "__main__":
    print(f"[1] {SCENARIO} 평가 서비스 초기화...")
    service = MockBenchmarkService(db_manager=None)

    print("\n[2] 정답 vs 휴리스틱 vs RL 비교...")
    service.evaluate_on_benchmark_dataset(benchmark_dataset=SCENARIO)
    print("\n[완료]")
