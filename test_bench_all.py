"""등록된 모든 벤치마크 시나리오 일괄 평가 (DB 불필요)."""
from biz.services.rl_scheduler_service import RLSchedulerService


if __name__ == "__main__":
    print("[1] 전체 벤치마크 평가...")
    service = RLSchedulerService(db_manager=None)
    service.evaluate_on_benchmark_datasets("all")
    print("\n[완료]")
