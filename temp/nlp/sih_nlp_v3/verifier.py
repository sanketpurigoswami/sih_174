"""
Protocol verifier / "brain" of the NLP side.

Input:
    expected task graph + normalized CV events

Output:
    CORRECT / UNCERTAIN / WRONG_OBJECT / WRONG_TARGET /
    REPEATED_STEP / SKIPPED_STEP / OUT_OF_ORDER / UNKNOWN

Important:
- It does not decide from one noisy frame.
- It consumes event-level CV output.
- It keeps protocol state.
"""

from typing import List, Optional
from .schemas import TaskSpec, CVEvent, VerificationResult

class ProtocolVerifier:
    def __init__(self, tasks: List[TaskSpec], min_confidence: float = 0.55):
        self.tasks = sorted(tasks, key=lambda x: x.step_id)
        self.min_confidence = min_confidence
        self.completed_steps = set()
        self.current_index = 0
        self.history = []

    @property
    def current_task(self) -> Optional[TaskSpec]:
        if self.current_index >= len(self.tasks):
            return None
        return self.tasks[self.current_index]

    def _match_task(self, task: TaskSpec, event: CVEvent) -> bool:
        if event.object and task.object and event.object != task.object:
            return False

        if event.target and task.target and event.target != task.target:
            return False

        expected_actions = {a.action for a in task.atomic_actions}
        if event.action not in expected_actions and event.action != task.high_level_action:
            return False

        return True

    def _object_matches(self, task, event):
        return not (event.object and task.object and event.object != task.object)

    def _target_matches(self, task, event):
        return not (event.target and task.target and event.target != task.target)

    def process(self, event: CVEvent) -> VerificationResult:
        if event.confidence < self.min_confidence:
            result = VerificationResult(
                status="UNCERTAIN",
                expected_step=self.current_task.step_id if self.current_task else None,
                observed_step=None,
                reason="CV confidence is below the verification threshold.",
                confidence=event.confidence,
                next_step=self.current_task.step_id if self.current_task else None
            )
            self.history.append(result)
            return result

        current = self.current_task
        if current is None:
            result = VerificationResult(
                status="REPEATED_STEP",
                expected_step=None,
                observed_step=None,
                reason="Experiment is already complete; a new event was observed.",
                confidence=event.confidence
            )
            self.history.append(result)
            return result

        # If the event belongs to a previously completed task.
        for done_id in self.completed_steps:
            done_task = next(t for t in self.tasks if t.step_id == done_id)
            if self._match_task(done_task, event):
                result = VerificationResult(
                    status="REPEATED_STEP",
                    expected_step=current.step_id,
                    observed_step=done_id,
                    reason=f"Step {done_id} appears to have been repeated.",
                    confidence=event.confidence,
                    next_step=current.step_id
                )
                self.history.append(result)
                return result

        # Exact match to current task.
        if self._match_task(current, event):
            # A high-level PLACE/INSERT event is treated as completion.
            if event.action in {current.high_level_action, "PLACE", "INSERT"}:
                self.completed_steps.add(current.step_id)
                self.current_index += 1
                next_id = self.current_task.step_id if self.current_task else None
                result = VerificationResult(
                    status="CORRECT",
                    expected_step=current.step_id,
                    observed_step=current.step_id,
                    reason=f"Step {current.step_id} completed correctly.",
                    confidence=event.confidence,
                    next_step=next_id
                )
            else:
                result = VerificationResult(
                    status="CORRECT",
                    expected_step=current.step_id,
                    observed_step=current.step_id,
                    reason=f"Expected atomic action {event.action} detected for step {current.step_id}.",
                    confidence=event.confidence,
                    next_step=current.step_id
                )
            self.history.append(result)
            return result

        # Correct step but wrong object/target/action.
        if not self._object_matches(current, event):
            result = VerificationResult(
                status="WRONG_OBJECT",
                expected_step=current.step_id,
                observed_step=None,
                reason=f"Expected {current.object}, observed {event.object}.",
                confidence=event.confidence,
                next_step=current.step_id
            )
            self.history.append(result)
            return result

        if not self._target_matches(current, event):
            result = VerificationResult(
                status="WRONG_TARGET",
                expected_step=current.step_id,
                observed_step=None,
                reason=f"Expected {current.target}, observed {event.target}.",
                confidence=event.confidence,
                next_step=current.step_id
            )
            self.history.append(result)
            return result

        # Search future steps for an out-of-order event.
        for future_index in range(self.current_index + 1, len(self.tasks)):
            future = self.tasks[future_index]
            if self._match_task(future, event):
                gap = [t.step_id for t in self.tasks[self.current_index:future_index]]
                result = VerificationResult(
                    status="OUT_OF_ORDER",
                    expected_step=current.step_id,
                    observed_step=future.step_id,
                    reason=f"Observed step {future.step_id} before completing step {current.step_id}.",
                    confidence=event.confidence,
                    next_step=current.step_id,
                    details={"skipped_expected_steps": gap}
                )
                self.history.append(result)
                return result

        result = VerificationResult(
            status="UNKNOWN",
            expected_step=current.step_id,
            observed_step=None,
            reason=f"Observed event {event.action} does not match the current protocol step.",
            confidence=event.confidence,
            next_step=current.step_id
        )
        self.history.append(result)
        return result
