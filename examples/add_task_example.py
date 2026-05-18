"""
[Example] 새로운 비즈니스 로직(서비스) 추가 가이드

이 예제는 '장비 점검' 기능을 새로운 서비스로 추가하는 과정을 보여줍니다.
"""

# STEP 1: biz/services/ 체크
# 실제 파일 경로: biz/services/maintenance_service.py (예시)
# ---------------------------------------------------------
# import logging
# from core.repository import BaseRepository
#
# logger = logging.getLogger(__name__)
#
# def handle_maintenance(repo: BaseRepository, equipment_id: str, params: dict):
#     logger.info(f"Adding maintenance record for {equipment_id}")
#     # DB 작업 예시
#     repo.insert("INSERT INTO maint_logs (eq_id, msg) VALUES (:id, :msg)", 
#                 {"id": equipment_id, "msg": params.get("message", "Scheduled")})
#     return True


# STEP 2: biz/task_processor.py 에 등록
# ---------------------------------------------------------
"""
# (위의 소스 상단에 임포트 추가)
from biz.services import common_service, rl_service, maintenance_service # <--- 새로 추가

class TaskProcessor:
    # ... 중략 ...
    def process(self, task: dict):
        # ... 중략 ...
        action = task.get('action')
        
        if action == "db_check":
            return common_service.handle_db_check(self.repo, process_id, params)
        elif action == "maintenance": # <--- 새로 추가된 분기문
            return maintenance_service.handle_maintenance(self.repo, equipment_id, params)
        # ... 후략 ...
"""

# STEP 3: API 호출 테스트
# ---------------------------------------------------------
import requests

def test_new_service():
    url = "http://127.0.0.1:8000/tasks"
    payload = {
        "equipment_id": "EQ_MAINT_001",
        "action": "maintenance",  # 등록한 action 명
        "parameters": {"message": "Monthly Checkup"}
    }
    # response = requests.post(url, json=payload)
    # print(response.json())

if __name__ == "__main__":
    print("이 코드는 새로운 작업을 시스템에 추가하는 방법을 설명하는 가이드입니다.")
    print("1. biz/services/ 폴더에 서비스 파일 작성")
    print("2. biz/task_processor.py 에 해당 서비스 임포트 및 action 등록")
    print("3. API(/tasks)를 통해 해당 action 호출")
