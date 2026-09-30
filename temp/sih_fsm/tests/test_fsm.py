import json
from pathlib import Path
from sih_fsm.normalizer import load_observed, load_procedure
from sih_fsm.fsm import ProtocolFSM

BASE = Path(__file__).resolve().parents[1]


def build_fsm():
    steps = load_procedure(str(BASE / "examples/task_metadata.json"))
    return ProtocolFSM(steps)


def ev(ts, action, obj="MOUSE", target="SLOT_1", conf=0.99):
    return {"timestamp": ts, "action": action, "object": obj, "target": target, "confidence": conf}


def test_successful_completion():
    fsm = build_fsm()
    rows = [ev(1,"REACH"), ev(2,"GRASP"), ev(3,"LIFT"), ev(4,"MOVE"), ev(5,"PLACE"), ev(6,"RELEASE")]
    results = [fsm.process(type("E", (), {})()) for _ in []]
    from sih_fsm.schemas import ObservedEvent
    results = [fsm.process(ObservedEvent.from_dict(x)) for x in rows]
    assert fsm.completed_steps == [1]
    assert results[-1].status == "CORRECT"
    assert results[-1].next_step == 2


def test_wrong_object():
    fsm = build_fsm()
    from sih_fsm.schemas import ObservedEvent
    r = fsm.process(ObservedEvent.from_dict(ev(1,"REACH",obj="EARBUDS_CASE",target="SLOT_2")))
    assert r.status == "WRONG_OBJECT" or r.status == "OUT_OF_ORDER"


def test_low_confidence():
    fsm = build_fsm()
    from sih_fsm.schemas import ObservedEvent
    r = fsm.process(ObservedEvent.from_dict(ev(1,"REACH",conf=0.2)))
    assert r.status == "UNCERTAIN"


def test_out_of_order():
    fsm = build_fsm()
    from sih_fsm.schemas import ObservedEvent
    r = fsm.process(ObservedEvent.from_dict(ev(1,"REACH",obj="EARBUDS_CASE",target="SLOT_2")))
    assert r.status == "OUT_OF_ORDER"
