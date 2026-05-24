from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from core.queue_manager import queue_manager
from core.config import config
import logging
from contextlib import asynccontextmanager

# Configure logging based on config
log_level = config.app.get('log_level', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)

# [Requirement 1] 인프라 구성: FastAPI 기반, Queue 관리
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start worker processes
    logger.info("Starting equipment transition scheduler workers...")
    try:
        queue_manager.start_workers()
        yield
    finally:
        # Shutdown: Stop worker processes
        logger.info("Stopping equipment transition scheduler workers...")
        queue_manager.stop_workers()

app = FastAPI(
    title="Equipment Transition Scheduler",
    description="FastAPI-based system for scheduling equipment transitions using a multi-processing queue.",
    lifespan=lifespan
)

class TaskRequest(BaseModel):
    """rule_timekey: 추론 Input 스냅샷(미지정·N/A 시 DB MAX). 학습 구간은 parameters.from/to_rule_timekey 사용."""
    rule_timekey: str = "N/A"
    action: str
    parameters: dict = {}

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Equipment Transition Scheduler API is running.",
        "worker_count": config.app.get('worker_processes', 2)
    }

@app.post("/tasks", status_code=202)
async def create_task(request: TaskRequest):
    """
    Endpoint to enqueue a new transition task.
    The task will be processed by workers in FIFO order.
    """
    try:
        task_data = request.model_dump() # model_dump() is preferred in Pydantic v2
        logger.info(f"Received task request: {task_data}")
        
        # [Requirement 1] Queue에 쌓인 요청에 대해 FIFO에 하나씩 꺼내어 프로세스에 할당한다.
        queue_manager.enqueue(task_data)
        
        return {
            "message": "Task enqueued successfully",
            "task": task_data
        }
    except Exception as e:
        logger.error(f"Failed to enqueue task: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    # In a real environment, this would be started via uvicorn CLI
    uvicorn.run(app, host="0.0.0.0", port=8000)
