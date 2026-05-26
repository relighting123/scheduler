"""벤치마크 데이터셋 평가 — DB에서 입력·정답 조회."""
from core.repository import BaseRepository
from biz.services.rl_scheduler_service import RLSchedulerService


def run_scenario(scenario: str, reload_seed: bool = False):
    print(f"\n{'=' * 60}\n[벤치마크] {scenario}\n{'=' * 60}")
    repo = BaseRepository()
    service = RLSchedulerService(db_manager=repo)
    service.ensure_rl_schema()
    service.seed_benchmark_scenarios(reload=reload_seed, scenarios=[scenario])
    service.evaluate_on_benchmark_dataset(benchmark_dataset=scenario)


if __name__ == "__main__":
    repo = BaseRepository()
    service = RLSchedulerService(db_manager=repo)
    service.ensure_rl_schema()
    scenarios = service.seed_benchmark_scenarios(reload=False)
    for scenario in scenarios:
        service.evaluate_on_benchmark_dataset(benchmark_dataset=scenario)
