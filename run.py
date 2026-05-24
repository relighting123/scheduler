import argparse
from core.repository import BaseRepository
from biz.services.rl_scheduler_service import RLSchedulerService

def main():
    parser = argparse.ArgumentParser(description="강화학습 스케줄러 간편 실행기 (CLI)")
    parser.add_argument("mode", choices=["train", "infer", "benchmark"], help="실행할 모드: 'train' (학습), 'infer' (추론), 또는 'benchmark' (조합최적화 비교)")
    parser.add_argument("--timekey", type=str, default=None, help="RULE_TIMEKEY (YYYYMMDDHHMMSS). 학습·추론 시 조회할 Input 스냅샷. 추론 Output 키로도 사용 (생략 시 DB 최신 또는 현재 시각)")
    parser.add_argument("--steps", type=int, default=100000, help="학습 시 진행할 총 타임스텝 (기본: 100000)")

    args = parser.parse_args()

    # DB 레포지토리 초기화 (실제 DB에 붙음)
    repo = BaseRepository()
    rl_service = RLSchedulerService(db_manager=repo)
    
    # Heuristic Trap 시나리오용 DB 재생성 및 데이터 삽입
    rl_service.init_db_scenario()

    if args.mode == "train":
        print(f"[학습 모드] RL 모델 학습을 시작합니다. (Timesteps: {args.steps})")
        rl_service.train_model(total_timesteps=args.steps, rule_timekey=args.timekey)
        print("학습이 완료되었습니다.")
        
    elif args.mode == "infer":
        print(f"[추론 모드] RL 모델 추론을 시작합니다. (Timekey: {args.timekey or '자동'})")
        results = rl_service.run_inference(rule_timekey=args.timekey)
        print(f"추론이 완료되었습니다. (결과 {len(results) if results else 0}건 도출)")

    elif args.mode == "benchmark":
        print(f"[벤치마크 모드] 조합최적화 스케줄링 벤치마크를 시작합니다. (Timesteps: {args.steps})")
        rl_service.run_combinatorial_benchmark(total_timesteps=args.steps)
        print("벤치마크가 완료되었습니다.")

if __name__ == "__main__":
    main()
