"""초기 전환 필수 벤치마크 — test/data/benchmark_initial_conv (DB 불필요)."""
from biz.services.rl_scheduler_service import RLSchedulerService
from biz.services.rl.test_data_loader import TestDataLoader


class MockBenchmarkService(RLSchedulerService):
    def init_benchmark_dataset_scenario(self):
        print("\n[Mock] 벤치마크 데이터셋 (test/data/benchmark_initial_conv)")

    def fetch_data(self, rule_timekey=None):
        loader = TestDataLoader()
        return loader.load_for_env("benchmark_initial_conv", rule_timekey=rule_timekey)


if __name__ == "__main__":
    print("[1] benchmark_initial_conv 평가 서비스 초기화...")
    service = MockBenchmarkService(db_manager=None)

    print("\n[2] 정답 vs 휴리스틱 vs RL 비교 (학습 생략, 기존 모델 있으면 RL 포함)...")
    service.evaluate_on_benchmark_dataset(
        benchmark_dataset="benchmark_initial_conv",
        model_path="scheduler_ppo_model",
    )
    print("\n[완료]")
