import json
from sih_fsm.normalizer import load_procedure, load_observed
from sih_fsm.fsm import ProtocolFSM

steps = load_procedure("examples/task_metadata.json")
events = load_observed("examples/cv_observed.jsonl")
fsm = ProtocolFSM(steps)

for event in events:
    result = fsm.process(event)
    print(json.dumps(result.to_dict(), indent=2))
