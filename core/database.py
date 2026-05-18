import oracledb
from core.config import config
from typing import Dict, Optional, Any, List, Union
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Core DB Manager for Oracle DB.
    Handles multiple DB connections via connection pools and provides
    low-level SQL execution helpers.
    """
    _instance = None
    _pools: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    def init_pools(self):
        """Initialize connection pools for all configured databases."""
        db_configs = config.databases
        for db_name, db_conf in db_configs.items():
            if db_name in self._pools:
                continue
            
            try:
                # Using thin mode by default (oracledb 2.0+)
                pool = oracledb.create_pool(
                    user=db_conf.get('user'),
                    password=db_conf.get('password'),
                    dsn=db_conf.get('dsn'),
                    min=1,
                    max=5,
                    increment=1
                )
                self._pools[db_name] = pool
                logger.info(f"Database pool for '{db_name}' initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize pool for '{db_name}': {e}")

    def get_connection(self, db_name: str = "primary"):
        """Acquire a connection from the specified pool."""
        if not self._pools:
            self.init_pools()
            
        pool = self._pools.get(db_name)
        if pool:
            return pool.acquire()
        raise ValueError(f"No pool found for database: {db_name}")

    def release_connection(self, conn, db_name: str = "primary"):
        """Release a connection back to the pool."""
        pool = self._pools.get(db_name)
        if pool:
            pool.release(conn)

    def close_all(self):
        """Close all connection pools."""
        for name, pool in self._pools.items():
            pool.close()
            logger.info(f"Database pool for '{name}' closed.")
        self._pools.clear()

    # --- SQL Execution Helpers ---

    def select(self, sql: str, params: Union[Dict, List, tuple] = None, db_name: str = "primary") -> List[Dict]:
        """Execute a SELECT statement and return results as a list of dictionaries."""
        conn = self.get_connection(db_name)
        try:
            with conn.cursor() as cursor:
                # Set rowfactory to return dictionaries
                cursor.rowfactory = lambda *args: dict(zip([d[0].lower() for d in cursor.description], args))
                cursor.execute(sql, params or [])
                return cursor.fetchall()
        finally:
            self.release_connection(conn, db_name)

    def execute(self, sql: str, params: Union[Dict, List, tuple] = None, db_name: str = "primary", commit: bool = True):
        """Execute non-query SQL (Insert, Update, Delete, Merge)."""
        conn = self.get_connection(db_name)
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, params or [])
                if commit:
                    conn.commit()
        except Exception as e:
            if commit:
                conn.rollback()
            raise e
        finally:
            self.release_connection(conn, db_name)

    def execute_many(self, sql: str, params_list: List[Union[Dict, List, tuple]], db_name: str = "primary", commit: bool = True):
        """Execute SQL for multiple parameter sets (Bulk operations)."""
        conn = self.get_connection(db_name)
        try:
            with conn.cursor() as cursor:
                cursor.executemany(sql, params_list)
                if commit:
                    conn.commit()
        except Exception as e:
            if commit:
                conn.rollback()
            raise e
        finally:
            self.release_connection(conn, db_name)

# Singleton instance
db_manager = DatabaseManager()
