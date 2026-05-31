"""
새 작업(action) 추가 예제 — 계획 배치 분석 서비스 기준
"""

# STEP 1: biz/services/ 에 handle_* 함수 정의
# STEP 2: biz/task_processor.py 에 action 분기 추가
# STEP 3: POST /tasks 호출

# payload 예시:
# {
#   "rule_timekey": "20251020070000",
#   "action": "plan_allocation",
#   "parameters": {
#     "scenario": "benchmark_dataset",
#     "optimize": true,
#     "max_iterations": 200
#   }
# }

if __name__ == "__main__":
    print("POST http://127.0.0.1:8000/tasks with action plan_allocation")
