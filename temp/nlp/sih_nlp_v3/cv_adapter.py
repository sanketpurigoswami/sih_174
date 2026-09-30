"""
Adapter for CV teammate output.

Your CV teammate should eventually send JSONL records such as:

{"timestamp":12.43,"action":"GRASP","object":"MOUSE",
 "object_id":7,"person_id":1,"hand":"right","confidence":0.93}

or:

{"timestamp":15.20,"action":"PLACE","object":"MOUSE",
 "target":"SLOT_1","object_id":7,"confidence":0.91}

This module normalizes spelling/casing and produces CVEvent objects.
"""

import json
from typing import Dict, Any, List
from .schemas import CVEvent

ACTION_ALIASES = {
    "grab": "GRASP",
    "grasp": "GRASP",
    "pick": "PICK_UP",
    "pickup": "PICK_UP",
    "pick_up": "PICK_UP",
    "lift": "LIFT",
    "move": "MOVE",
    "carry": "TRANSFER",
    "transfer": "TRANSFER",
    "put": "PLACE",
    "place": "PLACE",
    "set_down": "PLACE",
    "release": "RELEASE",
    "drop": "RELEASE",
    "touch": "TOUCH",
    "reach": "REACH",
    "hold": "HOLD",
    "insert": "INSERT",
    "remove": "REMOVE",
    "rotate": "ROTATE",
    "open": "OPEN",
    "close": "CLOSE",
    "operate": "OPERATE",
}

OBJECT_ALIASES = {
    "mouse": "MOUSE",
    "earbuds case": "EARBUDS_CASE",
    "earbud case": "EARBUDS_CASE",
    "orange box": "ORANGE_BOX",
    "white box": "WHITE_BOX",
    "pen": "PEN",
    "chocolate": "CHOCOLATE",
    "screwdriver": "SCREWDRIVER",
}

TARGET_ALIASES = {
    "slot 1": "SLOT_1",
    "slot 2": "SLOT_2",
    "slot 3": "SLOT_3",
    "slot 4": "SLOT_4",
    "table": "TABLE",
}

def normalize_event(raw: Dict[str, Any]) -> CVEvent:
    action_raw = str(raw.get("action", "UNKNOWN")).strip().lower()
    object_raw = str(raw.get("object", "")).strip().lower()
    target_raw = str(raw.get("target", "")).strip().lower()

    action = ACTION_ALIASES.get(action_raw, action_raw.upper())
    obj = OBJECT_ALIASES.get(object_raw, object_raw.upper() if object_raw else None)
    target = TARGET_ALIASES.get(target_raw, target_raw.upper() if target_raw else None)

    return CVEvent(
        timestamp=float(raw.get("timestamp", 0.0)),
        action=action,
        object=obj,
        target=target,
        object_id=raw.get("object_id"),
        person_id=raw.get("person_id"),
        hand=raw.get("hand"),
        confidence=float(raw.get("confidence", 1.0)),
        source=str(raw.get("source", "cv")),
    )

def load_jsonl(path: str) -> List[CVEvent]:
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(normalize_event(json.loads(line)))
    return sorted(events, key=lambda e: e.timestamp)
