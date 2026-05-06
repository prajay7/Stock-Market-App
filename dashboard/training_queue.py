from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

JobPayload = dict[str, Any]

QUEUE_LOCK = threading.Lock()
TRAINING_JOBS: dict[str, JobPayload] = {}
TRAINING_QUEUE: deque[str] = deque()
WORKER_THREAD: threading.Thread | None = None
WORKER_RUNNING = False
CHECKPOINT_PATH = Path("data/outputs/checkpoints/dashboard_training_queue.json")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def create_job(symbol: str, lookback_days: int, horizon_days: int, direct_internet_scrape: bool = False) -> JobPayload:
    return {
        "job_id": str(uuid.uuid4())[:8],
        "symbol": symbol,
        "lookback_days": int(lookback_days),
        "horizon_days": int(horizon_days),
        "direct_internet_scrape": bool(direct_internet_scrape),
        "status": "queued",
        "created_at": now_iso(),
        "started_at": None,
        "finished_at": None,
        "message": "Waiting in queue",
        "result": None,
    }


def enqueue_job(job: JobPayload) -> str:
    with QUEUE_LOCK:
        TRAINING_JOBS[job["job_id"]] = job
        TRAINING_QUEUE.append(job["job_id"])
        save_checkpoint()
    return job["job_id"]


def enqueue_jobs(jobs: list[JobPayload]) -> list[str]:
    job_ids = []
    with QUEUE_LOCK:
        for job in jobs:
            TRAINING_JOBS[job["job_id"]] = job
            TRAINING_QUEUE.append(job["job_id"])
            job_ids.append(job["job_id"])
        save_checkpoint()
    return job_ids


def get_jobs() -> list[JobPayload]:
    with QUEUE_LOCK:
        return [TRAINING_JOBS[job_id].copy() for job_id in list(TRAINING_QUEUE)] + [
            TRAINING_JOBS[job_id].copy() for job_id in TRAINING_JOBS if job_id not in TRAINING_QUEUE
        ]


def get_job(job_id: str) -> JobPayload | None:
    with QUEUE_LOCK:
        job = TRAINING_JOBS.get(job_id)
        return job.copy() if job else None


def has_running_worker() -> bool:
    global WORKER_THREAD
    return WORKER_THREAD is not None and WORKER_THREAD.is_alive()


def start_worker(process_fn: Callable[[JobPayload], None]) -> None:
    global WORKER_THREAD, WORKER_RUNNING
    if has_running_worker():
        return

    WORKER_RUNNING = True

    def _worker() -> None:
        global WORKER_RUNNING
        while True:
            with QUEUE_LOCK:
                if not TRAINING_QUEUE:
                    WORKER_RUNNING = False
                    break
                job_id = TRAINING_QUEUE[0]
                job = TRAINING_JOBS.get(job_id)
                if job and str(job.get("status")) == "canceled":
                    TRAINING_QUEUE.popleft()
                    save_checkpoint()
                    continue
                if job:
                    job["status"] = "running"
                    job["started_at"] = now_iso()
                    job["message"] = f"Training {job['symbol']}"
                    TRAINING_JOBS[job_id] = job
            if not job:
                with QUEUE_LOCK:
                    if TRAINING_QUEUE and TRAINING_QUEUE[0] == job_id:
                        TRAINING_QUEUE.popleft()
                continue

            try:
                process_fn(job)
            finally:
                with QUEUE_LOCK:
                    if TRAINING_QUEUE and TRAINING_QUEUE[0] == job_id:
                        TRAINING_QUEUE.popleft()
                    save_checkpoint()

    WORKER_THREAD = threading.Thread(target=_worker, daemon=True)
    WORKER_THREAD.start()


def update_job(job_id: str, **updates: Any) -> None:
    with QUEUE_LOCK:
        if job_id in TRAINING_JOBS:
            TRAINING_JOBS[job_id].update(updates)
            save_checkpoint()


def reset_queue() -> None:
    global WORKER_THREAD, WORKER_RUNNING
    with QUEUE_LOCK:
        TRAINING_JOBS.clear()
        TRAINING_QUEUE.clear()
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
    WORKER_THREAD = None
    WORKER_RUNNING = False


def cancel_jobs_by_symbols(symbols: list[str], reason: str = "Canceled by user") -> list[str]:
    global TRAINING_QUEUE
    canceled_job_ids: list[str] = []
    wanted = {str(symbol) for symbol in symbols}
    with QUEUE_LOCK:
        for job_id, job in TRAINING_JOBS.items():
            if str(job.get("symbol")) in wanted and str(job.get("status")) in {"queued", "running"}:
                job["status"] = "canceled"
                job["finished_at"] = now_iso()
                job["message"] = reason
                canceled_job_ids.append(job_id)
        TRAINING_QUEUE = deque([job_id for job_id in TRAINING_QUEUE if job_id not in canceled_job_ids])
        save_checkpoint()
    return canceled_job_ids


def save_checkpoint(path: Path = CHECKPOINT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "jobs": list(TRAINING_JOBS.values()),
        "queue": list(TRAINING_QUEUE),
        "saved_at": now_iso(),
        "running": WORKER_RUNNING,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def restore_checkpoint(path: Path = CHECKPOINT_PATH) -> None:
    if not path.exists():
        return

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return

    with QUEUE_LOCK:
        TRAINING_JOBS.clear()
        TRAINING_QUEUE.clear()
        for job in payload.get("jobs", []):
            if isinstance(job, dict) and job.get("job_id"):
                TRAINING_JOBS[str(job["job_id"])] = job
        for job_id in payload.get("queue", []):
            if job_id in TRAINING_JOBS:
                TRAINING_QUEUE.append(str(job_id))


def queue_status_for_symbol(symbol: str) -> str:
    with QUEUE_LOCK:
        jobs = [job for job in TRAINING_JOBS.values() if str(job.get("symbol")) == str(symbol)]

    if not jobs:
        return "Never Trained"

    def sort_key(job: JobPayload) -> str:
        return str(job.get("finished_at") or job.get("started_at") or job.get("created_at") or "")

    latest = sorted(jobs, key=sort_key)[-1]
    latest_status = str(latest.get("status", "")).lower()
    if latest_status in {"running", "queued"}:
        return "In Progress"
    if latest_status == "failed":
        return "Failed"
    if latest_status == "completed":
        return "Trained"
    return "Never Trained"


def latest_job_for_symbol(symbol: str) -> JobPayload | None:
    with QUEUE_LOCK:
        jobs = [job for job in TRAINING_JOBS.values() if str(job.get("symbol")) == str(symbol)]
    if not jobs:
        return None

    def sort_key(job: JobPayload) -> str:
        return str(job.get("finished_at") or job.get("started_at") or job.get("created_at") or "")

    return sorted(jobs, key=sort_key)[-1].copy()
