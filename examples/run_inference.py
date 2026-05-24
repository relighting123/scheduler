import os
import sys
import logging

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.repository import BaseRepository
from biz.services.rl_scheduler_service import RLSchedulerService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_rl_inference():
    """
    Loads a trained RL model and runs inference, saving the results to Excel.
    """
    logger.info("Initializing database and RL Scheduler Service for inference...")
    
    # Initialize BaseRepository (assuming 'primary' DB is configured)
    repo = BaseRepository(db_name="primary")
    
    # Initialize RLSchedulerService
    rl_service = RLSchedulerService(db_manager=repo)
    
    # Initialize a scenario in the database for inference
    # This will drop existing tables and insert sample data
    rl_service.init_db_scenario() 
    
    logger.info("Running RL inference...")
    # RULE_TIMEKEY: 미지정 시 WIP_INFO MAX — 조회·결과 출력 동일 키 사용
    rl_service.run_inference(rule_timekey=None)
    
    logger.info("RL inference completed. Check 'logs/simulation_logs/' for Excel output.")

if __name__ == "__main__":
    # Ensure the scheduler_ppo_model.zip exists for inference
    model_path = "scheduler_ppo_model.zip"
    if not os.path.exists(model_path):
        logger.warning(f"Model file '{model_path}' not found. "
                       "Please train a model first using 'rl_train' action "
                       "or ensure the model file is in the root directory.")
        # Optionally, you could call rl_service.train_model() here if you want to train if not found
        # For now, we'll just exit or proceed with random actions if model is not found.
        # The run_inference method already handles the case where the model is not found by using random actions.
    
    run_rl_inference()
