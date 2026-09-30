from typing import Dict, List, Tuple
from .fsm import ProtocolFSM
from .normalizer import load_observed, load_procedure
from .schemas import FSMResult


def run(procedure_path: str, observed_path: str, min_confidence: float = 0.55) -> Tuple[ProtocolFSM, List[FSMResult]]:
    steps = load_procedure(procedure_path)
    events = load_observed(observed_path)
    fsm = ProtocolFSM(steps, min_confidence=min_confidence)
    results = [fsm.process(event) for event in events]
    return fsm, results


def summary(fsm: ProtocolFSM, results: List[FSMResult]) -> Dict:
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return {
        "experiment_complete": fsm.is_complete,
        "completed_steps": list(fsm.completed_steps),
        "current_step": fsm.current_step.step_id if fsm.current_step else None,
        "current_action": fsm.current_action.action if fsm.current_action else None,
        "status_counts": counts,
        "events_processed": len(results),
    }
