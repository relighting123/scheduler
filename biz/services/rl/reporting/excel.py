"""시뮬레이션 결과 Excel·콘솔 출력 (logs/simulation_logs)."""

import os
from datetime import datetime

import pandas as pd

from biz.services.rl.reporting.allocation import build_allocation_pivot_df


def _simulation_log_path(file_name):
    log_dir = os.path.join(os.getcwd(), "logs", "simulation_logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, file_name)


def save_inference_summary(rule_timekey, allocation_df, achievement_df, file_prefix="inference_summary"):
    """추론·벤치마크 요약 리포트(할당 피벗 + 마지막 공정 달성률) 저장."""
    allocation_pivot_df = build_allocation_pivot_df(allocation_df)

    print("\n[1] 제품 공정별 장비모델별 대수 할당 결과")
    print("-" * 80)
    if allocation_pivot_df.empty:
        print("집계 가능한 장비 할당 결과가 없습니다.")
    else:
        print(allocation_pivot_df.to_string(index=False))
    print("-" * 80)

    print("\n[2] 마지막 공정 기준 제품별 계획달성률")
    print("-" * 80)
    if achievement_df.empty:
        print("집계 가능한 마지막 공정 계획달성률 정보가 없습니다.")
    else:
        print(achievement_df.to_string(index=False))
    print("-" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_timekey = str(rule_timekey).replace(" ", "_")
    file_path = _simulation_log_path(f"{file_prefix}_{safe_timekey}_{timestamp}.xlsx")

    with pd.ExcelWriter(file_path) as writer:
        allocation_pivot_df.to_excel(writer, sheet_name="EQP_ALLOCATION_PIVOT", index=False)
        allocation_df.to_excel(writer, sheet_name="EQP_ALLOCATION", index=False)
        achievement_df.to_excel(writer, sheet_name="LAST_OPER_ACH", index=False)

    print(f"[성공] 추론 요약 리포트가 엑셀로 저장되었습니다: {file_path}")


def save_production_logs(logs):
    """생산 로그 저장."""
    if not logs:
        return
    df = pd.DataFrame(logs)
    print(f"\n[시뮬레이션 생산 로그] 총 {len(df)}건")
    print("-" * 60)
    print(df.to_string(index=False))
    print("-" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = _simulation_log_path(f"production_log_{timestamp}.xlsx")
    df.to_excel(file_path, index=False)
    print(f"[성공] 생산 로그가 엑셀로 저장되었습니다: {file_path}")


def save_action_results(results):
    """장비 전환(RTD) 액션 로그 저장."""
    if not results:
        print("\n[장비 전환 액션] 도출된 전환 액션이 없습니다.")
        return

    df = pd.DataFrame(results)
    print(f"\n[장비 전환 액션] 총 {len(df)}건 도출")
    print("-" * 80)
    print(df.to_string(index=False))
    print("-" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = _simulation_log_path(f"action_log_{timestamp}.xlsx")
    df.to_excel(file_path, index=False)
    print(f"[성공] 액션 로그가 엑셀로 저장되었습니다: {file_path}")
