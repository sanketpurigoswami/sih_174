from dataclasses import dataclass, field
from pathlib import Path

ACTIONS = [
    "REACH", "TOUCH", "GRASP", "PICK_UP", "HOLD", "LIFT",
    "MOVE", "TRANSFER", "PLACE", "RELEASE", "OPEN", "CLOSE",
    "INSERT", "REMOVE", "ROTATE", "OPERATE"
]

OBJECTS = [
    "MOUSE", "EARBUDS_CASE", "ORANGE_BOX", "WHITE_BOX",
    "PEN", "CHOCOLATE", "SCREWDRIVER", "UNKNOWN_OBJECT"
]

TARGETS = [
    "SLOT_1", "SLOT_2", "SLOT_3", "SLOT_4", "TABLE",
    "UNKNOWN_TARGET"
]

@dataclass
class ModelConfig:
    # Downloaded once and then usable offline.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.42
    high_similarity_threshold: float = 0.62
    max_atomic_actions: int = 8

@dataclass
class Paths:
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    tasks_file: Path = field(init=False)
    output_dir: Path = field(init=False)

    def __post_init__(self):
        self.tasks_file = self.base_dir / "tasks.txt"
        self.output_dir = self.base_dir / "output"
        self.output_dir.mkdir(exist_ok=True)
