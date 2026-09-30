from dataclasses import replace
from typing import Dict, List, Optional, Tuple
from .schemas import ExpectedAction, FSMResult, ObservedEvent, ProcedureStep


class ProtocolFSM:
    """Deterministic protocol validator fed only semantic CV events."""

    STATUS_CORRECT = "CORRECT"
    STATUS_IN_PROGRESS = "IN_PROGRESS"
    STATUS_WRONG_OBJECT = "WRONG_OBJECT"
    STATUS_WRONG_TARGET = "WRONG_TARGET"
    STATUS_WRONG_ACTION = "WRONG_ACTION"
    STATUS_OUT_OF_ORDER = "OUT_OF_ORDER"
    STATUS_REPEATED_STEP = "REPEATED_STEP"
    STATUS_SKIPPED_STEP = "SKIPPED_STEP"
    STATUS_UNKNOWN = "UNKNOWN"
    STATUS_UNCERTAIN = "UNCERTAIN"
    STATUS_IGNORED = "IGNORED"
    STATUS_COMPLETED = "COMPLETED"

    def __init__(self, steps: List[ProcedureStep], min_confidence: float = 0.55,
                 repeat_window_s: float = 2.0, ignore_idle: bool = True):
        self.steps = sorted(steps, key=lambda s: s.step_id)
        self.min_confidence = min_confidence
        self.repeat_window_s = repeat_window_s
        self.ignore_idle = ignore_idle

        self.step_index = 0
        self.action_index = 0
        self.completed_steps: List[int] = []
        self.history: List[FSMResult] = []
        self.last_event: Optional[ObservedEvent] = None
        self.last_completed_timestamp: Optional[float] = None

    @property
    def current_step(self) -> Optional[ProcedureStep]:
        return self.steps[self.step_index] if self.step_index < len(self.steps) else None

    @property
    def current_action(self) -> Optional[ExpectedAction]:
        step = self.current_step
        if not step or not step.atomic_actions:
            return None
        return step.atomic_actions[self.action_index] if self.action_index < len(step.atomic_actions) else None

    @property
    def is_complete(self) -> bool:
        return self.step_index >= len(self.steps)

    def state_name(self) -> str:
        if self.is_complete:
            return "EXPERIMENT_COMPLETE"
        return f"STEP_{self.current_step.step_id}_{self.current_action.action if self.current_action else 'READY'}"

    def _event_confidence(self, event: ObservedEvent) -> float:
        values = [event.confidence]
        for x in (event.action_confidence, event.object_confidence, event.target_confidence):
            if x is not None:
                values.append(x)
        return min(values)

    @staticmethod
    def _norm(value: Optional[str]) -> Optional[str]:
        return value.strip().upper() if value else None

    def _same_obj(self, expected: Optional[str], observed: Optional[str]) -> bool:
        return expected is None or observed is None or self._norm(expected) == self._norm(observed)

    def _same_target(self, expected: Optional[str], observed: Optional[str]) -> bool:
        return expected is None or observed is None or self._norm(expected) == self._norm(observed)

    def _action_matches(self, expected: ExpectedAction, event: ObservedEvent) -> bool:
        return self._norm(expected.action) == self._norm(event.action)

    def _step_semantics_match(self, step: ProcedureStep, event: ObservedEvent) -> bool:
        action_match = any(self._norm(a.action) == self._norm(event.action) for a in step.atomic_actions)
        if step.high_level_action and self._norm(step.high_level_action) == self._norm(event.action):
            action_match = True
        return action_match and self._same_obj(step.object, event.object) and self._same_target(step.target, event.target)

    def _find_future_step(self, event: ObservedEvent) -> Optional[int]:
        for i in range(self.step_index + 1, len(self.steps)):
            if self._step_semantics_match(self.steps[i], event):
                return i
        return None

    def _find_completed_step(self, event: ObservedEvent) -> Optional[int]:
        done = set(self.completed_steps)
        for step in self.steps:
            if step.step_id not in done:
                continue
            if self._step_semantics_match(step, event):
                return step.step_id
        return None

    def _result(self, event: ObservedEvent, status: str, message: str,
                observed_step: Optional[int] = None, details: Optional[Dict] = None,
                expected_action: Optional[str] = None) -> FSMResult:
        current = self.current_step
        result = FSMResult(
            timestamp=event.timestamp,
            status=status,
            message=message,
            expected_step=current.step_id if current else None,
            observed_step=observed_step,
            expected_action=expected_action if expected_action is not None else (self.current_action.action if self.current_action else None),
            observed_action=event.action,
            next_step=current.step_id if current else None,
            next_action=self.current_action.action if self.current_action else None,
            confidence=self._event_confidence(event),
            state=self.state_name(),
            details=details or {},
        )
        self.history.append(result)
        self.last_event = event
        return result

    def process(self, event: ObservedEvent) -> FSMResult:
        confidence = self._event_confidence(event)

        if self.ignore_idle and event.action == "IDLE":
            return self._result(event, self.STATUS_IGNORED, "IDLE event ignored by protocol FSM.")

        if confidence < self.min_confidence:
            return self._result(event, self.STATUS_UNCERTAIN,
                                f"Observed event confidence {confidence:.2f} is below threshold {self.min_confidence:.2f}.")

        if self.is_complete:
            return self._result(event, self.STATUS_REPEATED_STEP,
                                "Experiment is complete; an additional protocol event was observed.",
                                details={"completed_steps": self.completed_steps})

        current = self.current_step
        expected = self.current_action
        assert current is not None

        # Repeated already-completed step gets priority over wrong-current diagnostics.
        repeated = self._find_completed_step(event)
        if repeated is not None:
            return self._result(event, self.STATUS_REPEATED_STEP,
                                f"Step {repeated} has already been completed and appears to be repeated.",
                                observed_step=repeated, details={"expected_current_step": current.step_id})

        # Explicit future-step detection catches out-of-order behavior.
        future_index = self._find_future_step(event)
        if future_index is not None:
            future_step = self.steps[future_index]
            skipped = [s.step_id for s in self.steps[self.step_index:future_index]]
            return self._result(
                event, self.STATUS_OUT_OF_ORDER,
                f"Observed step {future_step.step_id} before completing step {current.step_id}.",
                observed_step=future_step.step_id,
                details={"skipped_expected_steps": skipped},
            )

        # Object/target mismatches are actionable even when action is currently expected.
        if expected is not None:
            if not self._same_obj(expected.object or current.object, event.object):
                return self._result(event, self.STATUS_WRONG_OBJECT,
                                    f"Expected object {expected.object or current.object}, observed {event.object}.")
            if not self._same_target(expected.target or current.target, event.target):
                return self._result(event, self.STATUS_WRONG_TARGET,
                                    f"Expected target {expected.target or current.target}, observed {event.target}.")

        # Correct atomic action.
        if expected is not None and self._action_matches(expected, event):
            is_last = self.action_index == len(current.atomic_actions) - 1
            if is_last:
                return self._complete_current_step(event)

            self.action_index += 1
            next_action = self.current_action.action if self.current_action else None
            result = self._result(
                event, self.STATUS_CORRECT,
                f"Action {event.action} is correct for step {current.step_id}.",
                observed_step=current.step_id,
                details={"action_completed": expected.action, "action_index": self.action_index - 1},
            )
            # Override navigation after transition within same step.
            result.next_step = current.step_id
            result.next_action = next_action
            result.state = self.state_name()
            return result

        # The event has current step semantics but wrong atomic action.
        if self._step_semantics_match(current, event):
            return self._result(event, self.STATUS_WRONG_ACTION,
                                f"Expected action {expected.action if expected else 'UNKNOWN'}, observed {event.action}.",
                                observed_step=current.step_id)

        return self._result(event, self.STATUS_UNKNOWN,
                            f"Event {event.action} does not match the current protocol state.",
                            details={"current_step": current.step_id})

    def _completion_allowed(self, step: ProcedureStep) -> bool:
        condition = (step.completion_condition or "LAST_EXPECTED_ACTION").upper()
        if condition in {"LAST_EXPECTED_ACTION", "LAST_ACTION", "RELEASE", "PLACE_AND_RELEASE"}:
            return True
        return True

    def _complete_current_step(self, event: ObservedEvent) -> FSMResult:
        current = self.current_step
        assert current is not None
        step_id = current.step_id
        last_action = self.current_action.action if self.current_action else event.action

        self.completed_steps.append(step_id)
        self.step_index += 1
        self.action_index = 0
        self.last_completed_timestamp = event.timestamp

        if self.is_complete:
            next_step = None
            next_action = None
            state = "EXPERIMENT_COMPLETE"
            status = self.STATUS_COMPLETED
            msg = f"Step {step_id} completed. Experiment completed successfully."
        else:
            next_step = self.current_step.step_id
            next_action = self.current_action.action if self.current_action else None
            state = self.state_name()
            status = self.STATUS_CORRECT
            msg = f"Step {step_id} completed correctly. Next: step {next_step} ({next_action})."

        result = FSMResult(
            timestamp=event.timestamp,
            status=status,
            message=msg,
            expected_step=step_id,
            observed_step=step_id,
            expected_action=last_action,
            observed_action=event.action,
            next_step=next_step,
            next_action=next_action,
            confidence=self._event_confidence(event),
            state=state,
            details={"completed_step": step_id, "completed_steps": list(self.completed_steps)},
        )
        self.history.append(result)
        self.last_event = event
        return result

    def reset(self) -> None:
        self.step_index = 0
        self.action_index = 0
        self.completed_steps.clear()
        self.history.clear()
        self.last_event = None
        self.last_completed_timestamp = None
