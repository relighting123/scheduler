import argparse
from core.repository import BaseRepository
from biz.services.rl_scheduler_service import RLSchedulerService
from biz.services import plan_allocation_service

def main():
    parser = argparse.ArgumentParser(description="강화학습 스케줄러 간편 실행기 (CLI)")
    parser.add_argument(
        "mode",
        choices=["train", "infer", "benchmark", "plan-allocate"],
        help="실행 모드: train, infer, benchmark, plan-allocate(Input-only 장비 배치 분석)",
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
        help="?? ? validation ?? ??",
    )
    parser.add_argument(
        "--benchmark-dataset",
        type=str,
        default="benchmark_dataset",
        dest="benchmark_dataset",
        help="validation? ??? test/data ?? ???? ID (??: benchmark_dataset)",
    )

    parser.add_argument(
        "--single-benchmark",
        action="store_true",
        help="Evaluate only the scenario passed with --benchmark-dataset.",
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
        evaluate_all = not args.single_benchmark
        rl_service.train_model(
            total_timesteps=args.steps,
            rule_timekey=args.timekey,
            from_rule_timekey=args.from_timekey,
            to_rule_timekey=args.to_timekey,
            run_test_eval=not args.no_test_eval,
            benchmark_dataset=args.benchmark_dataset,
            evaluate_all_benchmarks=evaluate_all,
            benchmark_datasets=None if evaluate_all else [args.benchmark_dataset],
        )
        print("학습 및 벤치마크 데이터셋 평가가 완료되었습니다.")

    elif args.mode == "infer":
        resolved = args.timekey or "DB MAX(RULE_TIMEKEY)"
        print(f"[추론 모드] RULE_TIMEKEY={resolved} (입력·출력 동일)")
        results = rl_service.run_inference(rule_timekey=args.timekey)
        print(f"추론 완료 (전환 액션 {len(results) if results else 0}건)")

    elif args.mode == "benchmark":
        print("[벤치마크 모드] 저장된 모델 validation 평가")
        datasets = None if not args.single_benchmark else [args.benchmark_dataset]
        rl_service.run_benchmark_evaluation(datasets=datasets)
        print("벤치마크 데이터셋 평가 완료.")

    elif args.mode == "plan-allocate":
        print("[계획 배치 분석] Input-only 정적 최적화 (시뮬레이터 미사용)")
        plan_allocation_service.run_plan_allocation(
            repo,
            rule_timekey=args.timekey,
            scenario=args.benchmark_dataset,
            optimize=True,
        )
        print("계획 기반 장비 배치 분석이 완료되었습니다.")

if __name__ == "__main__":
    main()
