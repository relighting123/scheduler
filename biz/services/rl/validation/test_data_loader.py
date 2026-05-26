"""Deprecated: use BenchmarkDataAccess for DB-backed benchmark I/O.

CSV files under test/data are only used as the seed source for
`BenchmarkDataAccess.seed_all_from_directory()`.
"""
from biz.services.rl.validation.benchmark_data_access import (
    DATA_KEY_BY_FILE,
    DEFAULT_CSV_ROOT,
    INPUT_TABLE_FILES,
    BenchmarkDataAccess,
)

__all__ = [
    "BenchmarkDataAccess",
    "INPUT_TABLE_FILES",
    "DATA_KEY_BY_FILE",
    "DEFAULT_CSV_ROOT",
]
