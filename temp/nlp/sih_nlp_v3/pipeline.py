import json
from pathlib import Path
from typing import List, Dict, Any

from .transformer_nlp import TransformerNLP
from .task_graph import TaskGraphBuilder
from .cv_adapter import load_jsonl
from .verifier import ProtocolVerifier

class NLPV3Pipeline:
    def __init__(self, embedding_model=None):
        self.nlp = TransformerNLP()
        if embedding_model:
            self.nlp.config.embedding_model = embedding_model
        self.tasks = []
        self.graph = None
        self.verifier = None

    def build_from_tasks(self, tasks_file: str):
        self.tasks = self.nlp.parse_tasks_file(tasks_file)
        self.graph = TaskGraphBuilder().build(self.tasks)
        self.verifier = ProtocolVerifier(self.tasks)
        return self

    def export_metadata(self, output_dir: str):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "task_metadata.json", "w", encoding="utf-8") as f:
            json.dump([x.to_dict() for x in self.tasks], f, indent=2)

        with open(out / "task_graph.json", "w", encoding="utf-8") as f:
            json.dump(self.graph.to_dict(), f, indent=2)

        return out

    def verify_cv_jsonl(self, cv_jsonl: str, output_path: str = None):
        if self.verifier is None:
            raise RuntimeError("Call build_from_tasks() first.")

        events = load_jsonl(cv_jsonl)
        results = []

        for event in events:
            result = self.verifier.process(event)
            results.append({
                "timestamp": event.timestamp,
                "event": {
                    "action": event.action,
                    "object": event.object,
                    "target": event.target,
                    "object_id": event.object_id,
                    "person_id": event.person_id,
                    "hand": event.hand,
                    "confidence": event.confidence,
                },
                "verification": result.to_dict()
            })

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

        return results
