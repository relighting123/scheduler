import os
import logging
from core.repository import BaseRepository
from biz.services import common_service
from biz.services import plan_allocation_service

logger = logging.getLogger(__name__)


class TaskProcessor:
    """작업 큐에서 action별 비즈니스 서비스를 호출한다."""

    def __init__(self, repo: BaseRepository):
        self.repo = repo

    def process(self, task: dict):
        process_id = os.getpid()
        rule_timekey = task.get("rule_timekey", "N/A")
        action = task.get("action")
        params = task.get("parameters", {})

        logger.info(f"Processor (PID {process_id}) orchestrating service for '{action}'")

        if action == "db_check":
            return common_service.handle_db_check(self.repo, process_id, params)
        elif action in ("plan_allocation", "plan_optimize", "plan_allocation_optimize"):
            logger.info("Plan-based equipment allocation analysis (input-only)...")
            input_tk = params.get("rule_timekey") or (
                rule_timekey if rule_timekey not in ("N/A", "") else None
            )
            return plan_allocation_service.run_plan_allocation(
                self.repo,
                rule_timekey=input_tk,
                scenario=params.get("benchmark_dataset") or params.get("scenario"),
                optimize=params.get("optimize", True),
                include_marginal=params.get("include_marginal", True),
                max_iterations=int(params.get("max_iterations", 200)),
            )
        else:
            logger.warning(f"Unknown action: {action}")
            return False
