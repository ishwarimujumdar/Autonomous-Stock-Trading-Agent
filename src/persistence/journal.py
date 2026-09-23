import json
import os
from datetime import datetime, timezone

from src.config import JOURNAL_DIR


def _run_file(run_id: str) -> str:
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    return os.path.join(JOURNAL_DIR, f"{run_id}.jsonl")


def log_event(run_id: str, event_type: str, payload: dict) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "payload": payload,
    }
    with open(_run_file(run_id), "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
