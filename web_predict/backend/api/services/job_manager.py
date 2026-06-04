"""In-memory training job registry."""

from __future__ import annotations

import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TrainJob:
    job_id: str
    status: str = "pending"  # pending | running | completed | failed
    progress: int = 0
    epoch: int = 0
    total_epochs: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    message: str = ""
    model_path: str | None = None
    error: str | None = None
    task_type: str = "regression"
    best_metric: float | None = None
    metric_name: str = "val_loss"

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "message": self.message,
            "model_ready": bool(self.model_path),
            "error": self.error,
            "task_type": self.task_type,
            "best_metric": self.best_metric,
            "metric_name": self.metric_name,
        }


_jobs: dict[str, TrainJob] = {}
_lock = threading.Lock()


def create_job(task_type: str, total_epochs: int) -> TrainJob:
    job_id = str(uuid.uuid4())[:12]
    job = TrainJob(job_id=job_id, task_type=task_type, total_epochs=total_epochs)
    with _lock:
        _jobs[job_id] = job
    return job


def get_job(job_id: str) -> TrainJob | None:
    with _lock:
        job = _jobs.get(job_id)
        return deepcopy(job) if job else None


def update_job(job_id: str, **kwargs: Any) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        for k, v in kwargs.items():
            if hasattr(job, k):
                setattr(job, k, v)


def run_in_background(job_id: str, target: Callable[[], None]) -> None:
    def wrapper():
        update_job(job_id, status="running", message="训练进行中…")
        try:
            target()
            update_job(job_id, status="completed", progress=100, message="训练完成")
        except Exception as e:
            update_job(job_id, status="failed", error=str(e), message="训练失败")

    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
