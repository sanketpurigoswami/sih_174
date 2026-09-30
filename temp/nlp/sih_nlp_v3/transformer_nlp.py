"""
V3 Transformer semantic parser.

This is intentionally hybrid:
1. Lightweight rules extract explicit entities such as "Slot 1".
2. A pretrained Transformer encoder maps natural-language phrases to the
   canonical action/object ontology.
3. The result is converted into TaskSpec objects.

It does NOT need internet at inference time after the model has been
downloaded once.
"""

import re
from typing import Dict, List, Tuple, Optional

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

from .config import ACTIONS, OBJECTS, TARGETS, ModelConfig
from .schemas import TaskSpec, AtomicAction


ACTION_PHRASES = {
    "REACH": ["reach for", "reach toward", "move hand toward"],
    "TOUCH": ["touch", "tap"],
    "GRASP": ["grasp", "grab", "take hold of"],
    "PICK_UP": ["pick up", "pickup", "take", "lift up"],
    "HOLD": ["hold", "keep holding"],
    "LIFT": ["lift", "raise"],
    "MOVE": ["move", "move to", "shift"],
    "TRANSFER": ["transfer", "carry"],
    "PLACE": ["place", "put", "set down", "put down", "move into"],
    "RELEASE": ["release", "let go", "drop"],
    "OPEN": ["open"],
    "CLOSE": ["close", "shut"],
    "INSERT": ["insert", "put inside"],
    "REMOVE": ["remove", "take out"],
    "ROTATE": ["rotate", "turn"],
    "OPERATE": ["operate", "press", "activate", "use"],
}

OBJECT_PHRASES = {
    "MOUSE": ["mouse", "computer mouse"],
    "EARBUDS_CASE": ["earbuds case", "earbud case", "airpods case", "earphones case"],
    "ORANGE_BOX": ["orange box", "orange container"],
    "WHITE_BOX": ["white box", "white container"],
    "PEN": ["pen", "ball pen", "ballpoint pen"],
    "CHOCOLATE": ["chocolate", "chocolate bar"],
    "SCREWDRIVER": ["screwdriver", "driver"],
}

TARGET_PHRASES = {
    "SLOT_1": ["slot 1", "slot one", "position 1", "position one"],
    "SLOT_2": ["slot 2", "slot two", "position 2", "position two"],
    "SLOT_3": ["slot 3", "slot three", "position 3", "position three"],
    "SLOT_4": ["slot 4", "slot four", "position 4", "position four"],
    "TABLE": ["table", "work surface", "surface"],
}

class TransformerNLP:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self._action_embeddings = None
        self._object_embeddings = None
        self._target_embeddings = None

    def load(self):
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: "
                "pip install -r requirements.txt"
            )
        self.model = SentenceTransformer(self.config.embedding_model)
        self._action_embeddings = self.model.encode(
            list(ACTION_PHRASES.keys()), convert_to_tensor=True, normalize_embeddings=True
        )
        self._object_embeddings = self.model.encode(
            list(OBJECT_PHRASES.keys()), convert_to_tensor=True, normalize_embeddings=True
        )
        self._target_embeddings = self.model.encode(
            list(TARGET_PHRASES.keys()), convert_to_tensor=True, normalize_embeddings=True
        )
        return self

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _explicit_match(self, text: str, phrase_map: Dict[str, List[str]]) -> Optional[Tuple[str, float]]:
        text = self._clean(text)
        hits = []
        for canonical, phrases in phrase_map.items():
            for phrase in phrases:
                if phrase in text:
                    hits.append((canonical, 0.98, len(phrase)))
        if not hits:
            return None
        hits.sort(key=lambda x: x[2], reverse=True)
        return hits[0][0], hits[0][1]

    def _semantic_match(self, text: str, labels: List[str], embeddings):
        if self.model is None:
            self.load()
        query = self.model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(query, embeddings)[0]
        idx = int(scores.argmax())
        score = float(scores[idx])
        return labels[idx], score

    def _match(self, text: str, phrase_map: Dict[str, List[str]], embeddings):
        explicit = self._explicit_match(text, phrase_map)
        if explicit:
            return explicit
        labels = list(phrase_map.keys())
        label, score = self._semantic_match(text, labels, embeddings)
        return label, score

    def extract_step_id(self, text: str, fallback: int) -> int:
        m = re.search(r"(?:step|task)\s*[-:]?\s*(\d+)", text, re.I)
        return int(m.group(1)) if m else fallback

    def extract_target(self, text: str):
        explicit = self._explicit_match(text, TARGET_PHRASES)
        if explicit:
            return explicit

        # Only attempt semantic target matching if a target-like phrase exists.
        if re.search(r"\b(slot|position|surface|table)\b", text, re.I):
            return self._match(text, TARGET_PHRASES, self._target_embeddings)
        return None, 0.0

    def extract_object(self, text: str):
        explicit = self._explicit_match(text, OBJECT_PHRASES)
        if explicit:
            return explicit
        return self._match(text, OBJECT_PHRASES, self._object_embeddings)

    def extract_action(self, text: str):
        explicit = self._explicit_match(text, ACTION_PHRASES)
        if explicit:
            return explicit
        return self._match(text, ACTION_PHRASES, self._action_embeddings)

    def _atomic_plan(self, action: str, obj: Optional[str], target: Optional[str]):
        actions = []
        def add(a):
            actions.append(AtomicAction(
                action=a, object=obj, target=target,
                order=len(actions) + 1, confidence=1.0
            ))

        if action in {"PLACE", "TRANSFER"}:
            add("REACH")
            add("GRASP")
            add("LIFT")
            add("MOVE")
            add("PLACE")
            add("RELEASE")
        elif action in {"PICK_UP", "LIFT"}:
            add("REACH")
            add("GRASP")
            add(action)
        elif action == "GRASP":
            add("REACH")
            add("GRASP")
        elif action == "MOVE":
            add("GRASP")
            add("MOVE")
        else:
            add(action)

        return actions[:self.config.max_atomic_actions]

    def parse_instruction(self, text: str, step_number: int = 1) -> TaskSpec:
        text = text.strip()
        if not text:
            raise ValueError("Instruction cannot be empty.")

        step_id = self.extract_step_id(text, step_number)
        action, action_conf = self.extract_action(text)
        obj, obj_conf = self.extract_object(text)
        target, target_conf = self.extract_target(text)

        # For placement, PLACE is the canonical high-level action.
        if target and action in {"MOVE", "TRANSFER", "PICK_UP", "LIFT"}:
            if any(x in self._clean(text) for x in ["place", "put", "set down", "put down", "into"]):
                action = "PLACE"

        semantic_conf = min(action_conf, obj_conf, target_conf or action_conf)

        requirements = {
            "object_required": obj is not None,
            "target_required": target is not None,
            "hand_interaction": action in {
                "GRASP", "PICK_UP", "HOLD", "LIFT", "MOVE",
                "TRANSFER", "PLACE", "RELEASE", "INSERT", "REMOVE"
            },
            "completion_requires_target_contact": action in {"PLACE", "INSERT"},
            "completion_requires_object_motion": action in {
                "PICK_UP", "LIFT", "MOVE", "TRANSFER"
            },
            "allowed_observation_actions": [
                x.action for x in self._atomic_plan(action, obj, target)
            ],
        }

        return TaskSpec(
            step_id=step_id,
            source_text=text,
            high_level_action=action,
            object=obj,
            target=target,
            atomic_actions=self._atomic_plan(action, obj, target),
            requirements=requirements,
            semantic_confidence=semantic_conf,
        )

    def parse_tasks_file(self, path: str) -> List[TaskSpec]:
        specs = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "|" in line:
                    left, text = line.split("|", 1)
                    try:
                        step = int(left.strip())
                    except ValueError:
                        step = len(specs) + 1
                    specs.append(self.parse_instruction(text.strip(), step))
                else:
                    specs.append(self.parse_instruction(line, line_no))
        specs.sort(key=lambda x: x.step_id)
        return specs
