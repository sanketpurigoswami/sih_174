from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json

@dataclass
class Entity:
    value: str
    confidence: float = 1.0
    source: str = "nlp"

@dataclass
class AtomicAction:
    action: str
    object: Optional[str] = None
    target: Optional[str] = None
    order: int = 0
    confidence: float = 1.0

@dataclass
class TaskSpec:
    step_id: int
    source_text: str
    high_level_action: str
    object: Optional[str]
    target: Optional[str]
    atomic_actions: List[AtomicAction] = field(default_factory=list)
    requirements: Dict[str, Any] = field(default_factory=dict)
    semantic_confidence: float = 1.0
    metadata_version: str = "3.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

@dataclass
class TaskGraphNode:
    node_id: str
    step_id: int
    node_type: str
    action: str
    object: Optional[str]
    target: Optional[str]
    expected: bool = True

@dataclass
class TaskGraphEdge:
    source: str
    target: str
    relation: str
    condition: Optional[str] = None

@dataclass
class TaskGraph:
    nodes: List[TaskGraphNode]
    edges: List[TaskGraphEdge]
    start_node: str
    terminal_nodes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [asdict(x) for x in self.nodes],
            "edges": [asdict(x) for x in self.edges],
            "start_node": self.start_node,
            "terminal_nodes": self.terminal_nodes,
            "graph_version": "3.0"
        }

@dataclass
class CVEvent:
    timestamp: float
    action: str
    object: Optional[str] = None
    target: Optional[str] = None
    object_id: Optional[int] = None
    person_id: Optional[int] = None
    hand: Optional[str] = None
    confidence: float = 1.0
    source: str = "cv"

@dataclass
class VerificationResult:
    status: str
    expected_step: Optional[int]
    observed_step: Optional[int]
    reason: str
    confidence: float
    next_step: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
