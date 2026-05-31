from core.repository import BaseRepository
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_db_examples():
    """
    각 쿼리 항목별(Select, Insert, Update, Merge) 사용 예시입니다.
    개발자는 BaseRepository를 상속받거나 직접 호출하여 사용할 수 있습니다.
    """
    
    # [1] 리포지토리 초기화 (설정된 'primary' DB 사용)
    repo = BaseRepository(db_name="primary")

    # [2] SELECT (조회) 예시
    logger.info("--- SELECT 예시 ---")
    
    # 여러 행 조회 (select_list)
    users = repo.select_list(
        "SELECT * FROM users WHERE status = :status", 
        {"status": "ACTIVE"}
    )
    for user in users:
        print(f"User ID: {user['id']}, Name: {user['name']}")

    # 단일 행 조회 (select_one)
    user_detail = repo.select_one(
        "SELECT * FROM users WHERE id = :id", 
        {"id": "USER001"}
    )
    if user_detail:
        print(f"Detail: {user_detail['email']}")


    # [3] INSERT (저장) 예시
    logger.info("--- INSERT 예시 ---")
    repo.insert(
        "INSERT INTO users (id, name, email) VALUES (:id, :name, :email)",
        {"id": "USER002", "name": "Hong Gil-dong", "email": "hong@example.com"}
    )


    # [4] UPDATE (수정) 예시
    logger.info("--- UPDATE 예시 ---")
    repo.update(
        "UPDATE users SET name = :name WHERE id = :id",
        {"name": "Kim Chul-su", "id": "USER002"}
    )


    # [5] MERGE INTO 예시 (Oracle 고유 기능)
    logger.info("--- MERGE INTO 예시 ---")
    # 조건에 따라 UPDATE 또는 INSERT를 수행합니다.
    merge_sql = """
        MERGE INTO equipment_status t
        USING (SELECT :id AS eq_id, :status AS status FROM dual) s
        ON (t.eq_id = s.eq_id)
        WHEN MATCHED THEN
            UPDATE SET t.status = s.status, t.updated_at = SYSDATE
        WHEN NOT MATCHED THEN
            INSERT (t.eq_id, t.status, t.created_at)
            VALUES (s.eq_id, s.status, SYSDATE)
    """
    repo.merge(merge_sql, {"id": "EQ999", "status": "RUNNING"})


    # [6] 대량 처리 (BULK EXECUTE) 예시
    logger.info("--- 대량 INSERT 예시 ---")
    batch_data = [
        {"id": "LOG001", "msg": "Event 1"},
        {"id": "LOG002", "msg": "Event 2"},
        {"id": "LOG003", "msg": "Event 3"}
    ]
    repo.bulk_execute(
        "INSERT INTO event_logs (log_id, message) VALUES (:id, :msg)",
        batch_data
    )

if __name__ == "__main__":
    # 실제 실행을 위해서는 Oracle DB 연결 설정이 필요합니다.
    # 여기서는 코드 작성법(Usage)을 보여주는 것이 주 목적입니다.
    print("이 파일은 DB 조작 예시를 담고 있는 가이드 코드입니다.")
