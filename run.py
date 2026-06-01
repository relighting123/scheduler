import argparse
from core.repository import BaseRepository
from biz.services.rl_scheduler_service import RLSchedulerService
from biz.services import plan_allocation_service

def main():
    parser = argparse.ArgumentParser(
        description="장비 배치 CLI — allocate(정적·학습없음) / train·infer(RL)",
        epilog="전체 명령·API·옵션: COMMANDS.md 참고",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=[
            "allocate",
            "plan-allocate",
            "train-allocate",
            "infer-allocate",
            "train",
            "infer",
            "benchmark",
        ],
        help="train-allocate/infer-allocate=정적대수 RL, allocate=조합탐색, train/infer=시간대 RL",
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
        help="train: 학습 후 벤치마크 자동 평가 생략",
    )
    parser.add_argument(
        "--benchmark-dataset",
        type=str,
        default="benchmark_dataset",
        dest="benchmark_dataset",
        help="CSV 시나리오 ID (test/data/<ID>/). --use-db 없을 때 사용",
    )
    parser.add_argument(
        "--use-db",
        action="store_true",
        help="allocate/train-allocate/infer-allocate: Oracle RTS_LINEDSDB_INF 조회 (CSV 대신)",
    )

    parser.add_argument(
        "--single-benchmark",
        action="store_true",
        help="Evaluate only the scenario passed with --benchmark-dataset.",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="allocate: 현재 Input 배치만 평가 (재배치 탐색 생략)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="allocate: 상세 분석 리포트 추가 출력",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        dest="output_dir",
        help="allocate: JSON/CSV 저장 폴더 (기본 output/)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="static_allocation_ppo",
        dest="model_path",
        help="train-allocate/infer-allocate: PPO 모델 경로 (확장자 제외)",
    )
    parser.add_argument(
        "--no-bc",
        action="store_true",
        help="train-allocate: 모방학습(Expert) 사전학습 생략",
    )

    args = parser.parse_args()

    repo = BaseRepository()
    rl_service = RLSchedulerService(db_manager=repo)
    # CSV 시나리오 vs DB: --use-db 이면 InputDataAccess 경로
    data_scenario = None if args.use_db else args.benchmark_dataset

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

    elif args.mode in ("allocate", "plan-allocate"):
        src = "DB" if args.use_db else f"CSV({args.benchmark_dataset})"
        print(f"[정적 배치] 데이터 소스: {src}")
        plan_allocation_service.run_static_allocation(
            repo,
            rule_timekey=args.timekey,
            scenario=data_scenario,
            optimize=not args.no_optimize,
            verbose=args.verbose,
            output_dir=args.output_dir,
        )
        print("정적 배치 결과 생성 완료.")

    elif args.mode == "train-allocate":
        src = "DB" if args.use_db else f"CSV({args.benchmark_dataset})"
        print(f"[RL 정적 배치 학습] 데이터 소스: {src}")
        plan_allocation_service.run_rl_train(
            repo,
            rule_timekey=args.timekey,
            scenario=data_scenario,
            total_timesteps=args.steps,
            model_path=args.model_path,
            pretrain_bc=not args.no_bc,
        )
        print(f"학습 완료: {args.model_path}.zip")

    elif args.mode == "infer-allocate":
        tk = args.timekey or ("DB MAX(RULE_TIMEKEY)" if args.use_db else "CSV ground_truth")
        print(f"[RL 정적 배치 추론] RULE_TIMEKEY={tk} → 최종 대수표")
        plan_allocation_service.run_rl_infer(
            repo,
            rule_timekey=args.timekey,
            scenario=data_scenario,
            model_path=args.model_path,
            output_dir=args.output_dir,
        )
        print("RL 정적 배치 추론 완료.")

if __name__ == "__main__":
    main()
