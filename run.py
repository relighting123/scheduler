import argparse
from core.repository import BaseRepository
from biz.services import plan_allocation_service


def main():
    parser = argparse.ArgumentParser(description="계획 기반 장비 배치 분석 CLI")
    parser.add_argument(
        "--timekey",
        type=str,
        default=None,
        help="RULE_TIMEKEY (DB 조회 시, 미지정 시 MAX)",
    )
    parser.add_argument(
        "--benchmark-dataset",
        type=str,
        default="benchmark_dataset",
        dest="benchmark_dataset",
        help="test/data 시나리오 ID (지정 시 CSV 사용, DB 생략 가능)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="현재 배치만 분석 (재배치 탐색 생략)",
    )
    args = parser.parse_args()

    repo = BaseRepository()
    print("[계획 배치 분석] Input-only 정적 최적화")
    plan_allocation_service.run_plan_allocation(
        repo,
        rule_timekey=args.timekey,
        scenario=args.benchmark_dataset,
        optimize=not args.no_optimize,
    )
    print("완료.")


if __name__ == "__main__":
    main()
