import json
from typing import Any, Dict, List
from .schemas import ObservedEvent, ProcedureStep

ACTION_ALIASES = {
    "grab": "GRASP", "grasp": "GRASP",
    "pick": "PICK_UP", "pickup": "PICK_UP", "pick_up": "PICK_UP",
    "lift": "LIFT", "move": "MOVE", "carry": "TRANSFER", "transfer": "TRANSFER",
    "put": "PLACE", "place": "PLACE", "set_down": "PLACE", "put_down": "PLACE",
    "release": "RELEASE", "drop": "RELEASE",
    "reach": "REACH", "touch": "TOUCH", "hold": "HOLD",
    "insert": "INSERT", "remove": "REMOVE", "rotate": "ROTATE",
    "open": "OPEN", "close": "CLOSE", "operate": "OPERATE", "idle": "IDLE",
}


def normalize_action(value: Any) -> str:
    raw = str(value or "UNKNOWN").strip().lower()
    return ACTION_ALIASES.get(raw, raw.upper())


def load_observed(path: str) -> List[ObservedEvent]:
    """Load JSONL or a JSON array/object containing `events`."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    events: List[ObservedEvent] = []
    lines = [line for line in text.splitlines() if line.strip()]

    # JSONL is also a sequence of JSON objects, so detect it before trying
    # to parse the entire file as one JSON document.
    if len(lines) > 1:
        try:
            for line in lines:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL event must be an object")
                row["action"] = normalize_action(row.get("action"))
                events.append(ObservedEvent.from_dict(row))
        except json.JSONDecodeError:
            # Fall back to normal JSON parsing below if the file is not JSONL.
            events.clear()

    if not events:
        obj = json.loads(text)
        rows = obj.get("events", obj) if isinstance(obj, dict) else obj
        if isinstance(rows, dict):
            rows = [rows]
        for row in rows:
            row = dict(row)
            row["action"] = normalize_action(row.get("action"))
            events.append(ObservedEvent.from_dict(row))

    return sorted(events, key=lambda e: e.timestamp)


def load_procedure(path: str) -> List[ProcedureStep]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        rows = data.get("steps", data.get("tasks", data.get("procedure", [])))
    else:
        rows = data

    steps = [ProcedureStep.from_dict(x) for x in rows]
    if not steps:
        raise ValueError(f"No procedure steps found in {path}")
    steps.sort(key=lambda s: s.step_id)
    return steps
