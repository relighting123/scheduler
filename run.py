import argparse
from core.repository import BaseRepository
from biz.services.rl.benchmark_scenarios import DEFAULT_BENCHMARK_SCENARIO
from biz.services.rl_scheduler_service import RLSchedulerService

def main():
    parser = argparse.ArgumentParser(description="강화학습 스케줄러 간편 실행기 (CLI)")
    parser.add_argument(
        "mode",
        choices=["train", "infer", "benchmark"],
        help="실행 모드: train(학습), infer(추론), benchmark(벤치마크 데이터셋 평가)",
    )
    parser.add_argument(
        "--from-timekey",
        type=str,
        default=None,
        dest="from_timekey",
        help="학습 데이터 시작 RULE_TIMEKEY (YYYYMMDDHHMMSS). --to-timekey와 함께 구간 지정",
    )
    parser.add_argument(
        "--to-timekey",
        type=str,
        default=None,
        dest="to_timekey",
        help="학습 데이터 종료 RULE_TIMEKEY. from만 지정 시 단일 스냅샷",
    )
    parser.add_argument(
        "--timekey",
        type=str,
        default=None,
        help="RULE_TIMEKEY. 학습: 단일 스냅샷 / 추론: 조회·출력 공통 키 (미지정 시 DB MAX)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="학습·벤치마크 총 타임스텝 (기본: 100000)",
    )
    parser.add_argument(
        "--no-init-db",
        action="store_true",
        help="DB 시나리오 초기화(init_db_scenario) 생략",
    )
    parser.add_argument(
        "--no-test-eval",
        action="store_true",
        help="학습 후 벤치마크 데이터셋 성능 비교 생략",
    )
    parser.add_argument(
        "--benchmark-dataset",
        type=str,
        default=DEFAULT_BENCHMARK_SCENARIO,
        dest="benchmark_dataset",
        help=(
            "학습·벤치마크 평가 시나리오. 쉼표로 여러 개 또는 all "
            f"(기본: {DEFAULT_BENCHMARK_SCENARIO}; 예: bench_01,bench_02 또는 all)"
        ),
    )

    args = parser.parse_args()

    repo = BaseRepository()
    rl_service = RLSchedulerService(db_manager=repo)

    if not args.no_init_db and args.mode in ("train", "infer"):
        rl_service.init_db_scenario()

    if args.mode == "train":
        print(f"[학습 모드] RL 학습 (Timesteps: {args.steps})")
        if args.from_timekey or args.to_timekey:
            print(
                f"  학습 구간: {args.from_timekey or args.to_timekey} "
                f"~ {args.to_timekey or args.from_timekey}"
            )
        elif args.timekey:
            print(f"  학습 스냅샷: {args.timekey}")
        else:
            print("  학습 스냅샷: DB MAX(RULE_TIMEKEY) 또는 기본값")
        rl_service.train_model(
            total_timesteps=args.steps,
            rule_timekey=args.timekey,
            from_rule_timekey=args.from_timekey,
            to_rule_timekey=args.to_timekey,
            run_test_eval=not args.no_test_eval,
            benchmark_dataset=args.benchmark_dataset,
        )
        print("학습 및 벤치마크 데이터셋 평가가 완료되었습니다.")

    elif args.mode == "infer":
        resolved = args.timekey or "DB MAX(RULE_TIMEKEY)"
        print(f"[추론 모드] RULE_TIMEKEY={resolved} (입력·출력 동일)")
        results = rl_service.run_inference(rule_timekey=args.timekey)
        print(f"추론 완료 (전환 액션 {len(results) if results else 0}건)")

    elif args.mode == "benchmark":
        from biz.services.rl.benchmark_scenarios import parse_benchmark_scenarios

        scenario_ids = parse_benchmark_scenarios(args.benchmark_dataset)
        print(f"[벤치마크 모드] 시나리오={scenario_ids} (파일 기반 평가)")
        rl_service.evaluate_on_benchmark_datasets(
            benchmark_datasets=args.benchmark_dataset
        )
        print("벤치마크 시나리오 평가 완료.")

if __name__ == "__main__":
    main()
