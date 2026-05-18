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
            # 파라미터로 학습 스텝 등 전달 가능
            timesteps = params.get("total_timesteps", 10000)
            rl_service.train_model(total_timesteps=timesteps)
            return True
        elif action == "rl_inference":
            logger.info(f"Starting RL Inference for Rule Timekey: {rule_timekey}...")
            rl_service = RLSchedulerService(db_manager=self.repo)
            results = rl_service.run_inference(rule_timekey=rule_timekey)
            return True
        elif action == "combinatorial_benchmark":
            logger.info("Starting Combinatorial Optimization Benchmark...")
            rl_service = RLSchedulerService(db_manager=self.repo)
            timesteps = params.get("total_timesteps", 10000)
            rl_service.run_combinatorial_benchmark(total_timesteps=timesteps)
            return True
        else:
            logger.warning(f"Unknown action: {action}")
            return False

