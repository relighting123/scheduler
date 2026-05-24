"""조합최적화 벤치마크 — test/data 시나리오 로더 사용 (DB 불필요)."""
from biz.services.rl_scheduler_service import RLSchedulerService
from biz.services.rl.test_data_loader import TestDataLoader


class TestCombinatorialBenchmarkService(RLSchedulerService):
    def init_combinatorial_scenario(self):
        print("\n[DB 설정] 모의(Mock) 조합최적화 벤치마크 (test/data/combinatorial)")

    def fetch_data(self, rule_timekey=None):
        loader = TestDataLoader()
        return loader.load_for_env("combinatorial", rule_timekey=rule_timekey)


if __name__ == "__main__":
    print("[1단계] test/data 기반 벤치마크 서비스 초기화...")
    test_service = TestCombinatorialBenchmarkService(db_manager=None)

    print("\n[2단계] 조합최적화 벤치마크 (정답 vs 휴리스틱 vs RL 학습+비교)...")
    test_service.run_combinatorial_benchmark(total_timesteps=60000)
    print("\n[성공] 벤치마크 완료.")
