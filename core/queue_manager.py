import multiprocessing
import os
import time
import logging
from typing import Any, List
from core.config import config
from core.database import db_manager
from core.repository import BaseRepository

from biz.task_processor import TaskProcessor

# Note: Logging in multi-processing can be tricky. 
# For simplicity, we use basic logging here.
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (%(process)d) %(message)s')
logger = logging.getLogger(__name__)

class Worker(multiprocessing.Process):
    """
    Independent worker process that consumes tasks from a FIFO queue.
    """
    def __init__(self, queue: multiprocessing.Queue):
        super().__init__()
        self.queue = queue
        self.daemon = True

    def run(self):
        """
        Main loop for the worker process.
        Initializes DB connections within the process to ensure isolation.
        """
        process_id = os.getpid()
        logger.info(f"Worker process {process_id} starting...")
        
        try:
            db_manager.init_pools()
            # Initialize repository and processor for this process
            repo = BaseRepository(db_name="primary")
            self.processor = TaskProcessor(repo)
        except Exception as e:
            logger.error(f"Worker {process_id} failed to initialize DB/Repo/Processor: {e}")

        while True:
            try:
                task = self.queue.get()
                
                if task == "STOP":
                    logger.info(f"Worker {process_id} received STOP signal.")
                    break
                
                self.process_task(task)
            except Exception as e:
                logger.error(f"Worker {process_id} encountered an error: {e}")
            except KeyboardInterrupt:
                break

        db_manager.close_all()
        logger.info(f"Worker {process_id} finished.")

    def process_task(self, task: Any):
        """
        Delegates the actual business logic to the TaskProcessor.
        """
        try:
            self.processor.process(task)
        except Exception as e:
            logger.error(f"Error in task delegation: {e}")

class QueueManager:
    """
    Core manager for multi-processing queue and worker rotation.
    """
    _instance = None
    _queue = multiprocessing.Queue()
    _workers: List[Worker] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QueueManager, cls).__new__(cls)
        return cls._instance

    def start_workers(self):
        """Spawn worker processes based on configuration."""
        if self._workers:
            logger.warning("Workers are already running.")
            return

        num_workers = config.app.get('worker_processes', 2)
        logger.info(f"Starting {num_workers} worker processes...")
        
        for i in range(num_workers):
            worker = Worker(self._queue)
            worker.start()
            self._workers.append(worker)

    def enqueue(self, task: Any):
        """Put a task into the FIFO queue."""
        self._queue.put(task)

    def stop_workers(self):
        """Gracefully stop all worker processes."""
        logger.info("Stopping all workers...")
        for _ in range(len(self._workers)):
            self._queue.put("STOP")
        
        for worker in self._workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
        
        self._workers.clear()
        logger.info("All workers stopped.")

# Singleton instance
queue_manager = QueueManager()
