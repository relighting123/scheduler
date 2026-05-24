"""벤치마크 데이터셋 평가 — test/data/benchmark_dataset 로더 사용 (DB 불필요)."""
from biz.services.rl_scheduler_service import RLSchedulerService
from biz.services.rl.test_data_loader import TestDataLoader


class MockBenchmarkDatasetService(RLSchedulerService):
    def init_benchmark_dataset_scenario(self):
        print("\n[Mock] 벤치마크 데이터셋 (test/data/benchmark_dataset)")

    def fetch_data(self, rule_timekey=None):
        loader = TestDataLoader()
        return loader.load_for_env("benchmark_dataset", rule_timekey=rule_timekey)


if __name__ == "__main__":
    print("[1] 벤치마크 데이터셋 평가 서비스 초기화...")
    service = MockBenchmarkDatasetService(db_manager=None)

    print("\n[2] 벤치마크 데이터셋 평가 (정답 vs 휴리스틱 vs RL)...")
    service.run_benchmark_evaluation(total_timesteps=60000)
    print("\n[완료]")
