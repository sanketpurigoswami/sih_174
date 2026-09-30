from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ExpectedAction:
    action: str
    object: Optional[str] = None
    target: Optional[str] = None
    order: int = 0


@dataclass
class ProcedureStep:
    step_id: int
    source_text: str = ""
    high_level_action: Optional[str] = None
    object: Optional[str] = None
    target: Optional[str] = None
    atomic_actions: List[ExpectedAction] = field(default_factory=list)
    completion_condition: str = "LAST_EXPECTED_ACTION"
    semantic_confidence: float = 1.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProcedureStep":
        raw_actions = d.get("atomic_actions") or d.get("expected_actions") or []
        actions: List[ExpectedAction] = []
        for i, a in enumerate(raw_actions, 1):
            if isinstance(a, str):
                actions.append(ExpectedAction(action=a.upper(), order=i,
                                              object=d.get("object"), target=d.get("target")))
            else:
                actions.append(ExpectedAction(
                    action=str(a.get("action", "UNKNOWN")).upper(),
                    object=a.get("object", d.get("object")),
                    target=a.get("target", d.get("target")),
                    order=int(a.get("order", i)),
                ))
        return cls(
            step_id=int(d["step_id"]),
            source_text=d.get("source_text", ""),
            high_level_action=d.get("high_level_action"),
            object=d.get("object"),
            target=d.get("target"),
            atomic_actions=actions,
            completion_condition=d.get("completion_condition", "LAST_EXPECTED_ACTION"),
            semantic_confidence=float(d.get("semantic_confidence", 1.0)),
        )


@dataclass
class ObservedEvent:
    timestamp: float
    action: str
    object: Optional[str] = None
    target: Optional[str] = None
    confidence: float = 1.0
    object_confidence: Optional[float] = None
    action_confidence: Optional[float] = None
    target_confidence: Optional[float] = None
    object_id: Optional[int] = None
    person_id: Optional[int] = None
    hand: Optional[str] = None
    source: str = "cv"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ObservedEvent":
        return cls(
            timestamp=float(d.get("timestamp", 0.0)),
            action=str(d.get("action", "UNKNOWN")).strip().upper(),
            object=_norm_optional(d.get("object")),
            target=_norm_optional(d.get("target")),
            confidence=float(d.get("confidence", 1.0)),
            object_confidence=_float_optional(d.get("object_confidence")),
            action_confidence=_float_optional(d.get("action_confidence")),
            target_confidence=_float_optional(d.get("target_confidence")),
            object_id=d.get("object_id"),
            person_id=d.get("person_id"),
            hand=d.get("hand"),
            source=str(d.get("source", "cv")),
        )


@dataclass
class FSMResult:
    timestamp: float
    status: str
    message: str
    expected_step: Optional[int]
    observed_step: Optional[int]
    expected_action: Optional[str]
    observed_action: Optional[str]
    next_step: Optional[int]
    next_action: Optional[str]
    confidence: float
    state: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _norm_optional(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value.upper() if value else None


def _float_optional(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)
