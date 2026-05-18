from typing import List, Dict, Any, Optional
from core.database import db_manager

class BaseRepository:
    """
    Base Repository that provides developer-friendly methods for DB operations.
    Developers can use this as-is or inherit from it to implement specific logic.
    [Requirement 2] 이러한 DB 연결 방식등 역시 별도 Core 형태로 관리하며 사용자는 사용법만 알면 가져다 쓰게 한다.
    """
    def __init__(self, db_name: str = "primary"):
        self.db_name = db_name

    def select_list(self, sql: str, params: Any = None) -> List[Dict]:
        """조회: 여러 행 반환"""
        return db_manager.select(sql, params, self.db_name)

    def select_one(self, sql: str, params: Any = None) -> Optional[Dict]:
        """조회: 단일 행 반환"""
        results = self.select_list(sql, params)
        return results[0] if results else None

    def insert(self, sql: str, params: Any = None):
        """저장: INSERT"""
        return db_manager.execute(sql, params, self.db_name)

    def execute(self, sql: str, params: Any = None):
        """범용 실행: DDL(DROP, CREATE) 등"""
        return db_manager.execute(sql, params, self.db_name)

    def update(self, sql: str, params: Any = None):
        """업데이트: UPDATE"""
        return db_manager.execute(sql, params, self.db_name)

    def merge(self, sql: str, params: Any = None):
        """MERGE INTO"""
        return db_manager.execute(sql, params, self.db_name)

    def bulk_execute(self, sql: str, params_list: List[Any]):
        """대량 처리: EXECUTEMANY"""
        return db_manager.execute_many(sql, params_list, self.db_name)
