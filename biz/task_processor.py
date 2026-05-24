import os
import logging
from core.repository import BaseRepository
from biz.services import common_service
from biz.services.rl_scheduler_service import RLSchedulerService

logger = logging.getLogger(__name__)

class TaskProcessor:
    """
    Orchestrator for task processing.
    Delegates work to specific services in the biz.services package.
    """
    def __init__(self, repo: BaseRepository):
        self.repo = repo

    def process(self, task: dict):
        """
        Main entry point for task processing.
        """
        process_id = os.getpid()
        rule_timekey = task.get('rule_timekey', 'N/A')
        action = task.get('action')
        params = task.get('parameters', {})

        logger.info(f"Processor (PID {process_id}) orchestrating service for '{action}'")

        if action == "db_check":
            return common_service.handle_db_check(self.repo, process_id, params)
        elif action == "transition":
            return common_service.handle_transition(self.repo, rule_timekey, action, params)
        elif action == "rl_train":
            logger.info("Starting RL Training...")
            rl_service = RLSchedulerService(db_manager=self.repo)
            timesteps = params.get("total_timesteps", 10000)
            from_tk = params.get("from_rule_timekey") or params.get("from_timekey")
            to_tk = params.get("to_rule_timekey") or params.get("to_timekey")
            single_tk = params.get("rule_timekey") or (
                rule_timekey if rule_timekey not in ("N/A", "") else None
            )
            run_test = params.get("run_test_eval", True)
            benchmark_dataset = params.get("benchmark_dataset", "benchmark_dataset")
            rl_service.train_model(
                total_timesteps=timesteps,
                rule_timekey=single_tk,
                from_rule_timekey=from_tk,
                to_rule_timekey=to_tk,
                run_test_eval=run_test,
                benchmark_dataset=benchmark_dataset,
            )
            return True
        elif action == "rl_inference":
            logger.info("Starting RL Inference...")
            rl_service = RLSchedulerService(db_manager=self.repo)
            input_tk = params.get("rule_timekey") or (
                rule_timekey if rule_timekey not in ("N/A", "") else None
            )
            rl_service.run_inference(rule_timekey=input_tk)
            return True
        elif action in ("benchmark_evaluation", "benchmark"):
            logger.info("Starting benchmark dataset evaluation...")
            rl_service = RLSchedulerService(db_manager=self.repo)
            timesteps = params.get("total_timesteps", 10000)
            rl_service.run_benchmark_evaluation(total_timesteps=timesteps)
            return True
        else:
            logger.warning(f"Unknown action: {action}")
            return False

