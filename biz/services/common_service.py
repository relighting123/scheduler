import time
import logging
from core.repository import BaseRepository

logger = logging.getLogger(__name__)

def handle_db_check(repo: BaseRepository, pid: int, params: dict):
    """Example task: Parallel SELECT 1"""
    logger.info(f"Worker {pid} performing DB check via CommonHandler...")
    result = repo.select_one("SELECT 1 AS val FROM DUAL")
    # Artificial delay
    time.sleep(2)
    logger.info(f"Worker {pid} DB check result: {result}")
    return True

def handle_transition(repo: BaseRepository, equipment_id: str, action: str, params: dict):
    """Example task: Equipment status transition"""
    logger.info(f"Handling transition for {equipment_id} via CommonHandler")
    current_status = repo.select_one(
        "SELECT * FROM equipment_status WHERE eq_id = :id", 
        {"id": equipment_id}
    )
    time.sleep(0.5)

    merge_sql = """
        MERGE INTO equipment_history h
        USING (SELECT :id as id, :act as act FROM dual) s
        ON (h.eq_id = s.id)
        WHEN MATCHED THEN UPDATE SET h.status = s.act, h.updated_at = SYSDATE
        WHEN NOT MATCHED THEN INSERT (h.eq_id, h.status, h.created_at) VALUES (s.id, s.act, SYSDATE)
    """
    repo.merge(merge_sql, {"id": equipment_id, "act": action})
    return True
