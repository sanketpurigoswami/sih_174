import csv
import json
from pathlib import Path
from typing import Iterable, List
from .schemas import FSMResult


def write_results_json(results: Iterable[FSMResult], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)


def write_results_jsonl(results: Iterable[FSMResult], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")


def write_results_csv(results: Iterable[FSMResult], path: str) -> None:
    rows = [r.to_dict() for r in results]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = [
        "timestamp", "status", "message", "expected_step", "observed_step",
        "expected_action", "observed_action", "next_step", "next_action",
        "confidence", "state", "details"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["details"] = json.dumps(row.get("details", {}), separators=(",", ":"))
            writer.writerow({k: row.get(k) for k in fields})
