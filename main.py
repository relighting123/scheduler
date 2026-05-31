from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.queue_manager import queue_manager
from core.config import config
import logging
from contextlib import asynccontextmanager

log_level = config.app.get("log_level", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting plan allocation workers...")
    try:
        queue_manager.start_workers()
        yield
    finally:
        logger.info("Stopping workers...")
        queue_manager.stop_workers()


app = FastAPI(
    title="Plan Allocation Analyzer",
    description="Input-only static plan achievement and equipment allocation analysis.",
    lifespan=lifespan,
)


class TaskRequest(BaseModel):
    rule_timekey: str = "N/A"
    action: str
    parameters: dict = {}


@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Plan allocation API is running.",
        "worker_count": config.app.get("worker_processes", 2),
    }


@app.post("/tasks", status_code=202)
async def create_task(request: TaskRequest):
    try:
        task_data = request.model_dump()
        logger.info(f"Received task request: {task_data}")
        queue_manager.enqueue(task_data)
        return {"message": "Task enqueued successfully", "task": task_data}
    except Exception as e:
        logger.error(f"Failed to enqueue task: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
