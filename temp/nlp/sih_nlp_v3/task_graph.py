from typing import List
from .schemas import TaskSpec, TaskGraph, TaskGraphNode, TaskGraphEdge

class TaskGraphBuilder:
    """
    Converts TaskSpec objects into a directed protocol graph.

    Graph structure:
        START -> step1.atomic1 -> ... -> step1.atomicN
              -> step2.atomic1 -> ... -> ...
              -> END

    The graph also contains STEP_COMPLETE nodes. These become useful when
    CV emits high-level events rather than every atomic hand movement.
    """

    def build(self, tasks: List[TaskSpec]) -> TaskGraph:
        if not tasks:
            raise ValueError("At least one task is required.")

        nodes = [
            TaskGraphNode(
                node_id="START", step_id=0, node_type="START",
                action="START", object=None, target=None
            )
        ]
        edges = []
        previous = "START"

        for task in tasks:
            step_root = f"S{task.step_id}_START"
            nodes.append(TaskGraphNode(
                node_id=step_root,
                step_id=task.step_id,
                node_type="STEP_START",
                action="STEP_START",
                object=task.object,
                target=task.target
            ))
            edges.append(TaskGraphEdge(previous, step_root, "NEXT_STEP"))

            atomic_ids = []
            for atomic in task.atomic_actions:
                node_id = f"S{task.step_id}_A{atomic.order}_{atomic.action}"
                atomic_ids.append(node_id)
                nodes.append(TaskGraphNode(
                    node_id=node_id,
                    step_id=task.step_id,
                    node_type="ATOMIC_ACTION",
                    action=atomic.action,
                    object=atomic.object,
                    target=atomic.target
                ))

            complete_id = f"S{task.step_id}_COMPLETE"
            nodes.append(TaskGraphNode(
                node_id=complete_id,
                step_id=task.step_id,
                node_type="STEP_COMPLETE",
                action="STEP_COMPLETE",
                object=task.object,
                target=task.target
            ))

            if atomic_ids:
                edges.append(TaskGraphEdge(step_root, atomic_ids[0], "EXPECTED"))
                for a, b in zip(atomic_ids, atomic_ids[1:]):
                    edges.append(TaskGraphEdge(a, b, "EXPECTED"))
                edges.append(TaskGraphEdge(atomic_ids[-1], complete_id, "COMPLETE"))

            previous = complete_id

        terminal = "END"
        nodes.append(TaskGraphNode(
            node_id=terminal, step_id=tasks[-1].step_id,
            node_type="END", action="END", object=None, target=None
        ))
        edges.append(TaskGraphEdge(previous, terminal, "TERMINAL"))

        return TaskGraph(
            nodes=nodes,
            edges=edges,
            start_node="START",
            terminal_nodes=[terminal]
        )
