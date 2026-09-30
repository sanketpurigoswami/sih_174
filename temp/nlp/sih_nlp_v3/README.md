# SIH PS 26174 — NLP V3

This is the complete V3 NLP/protocol module intended to sit between:

    English experiment procedure
              ↓
       NLP V3 / Task Graph
              ↓
       expected protocol
              ↑
       CV/HAR structured events
              ↓
          verifier
              ↓
     correct / anomaly / next step

## What V3 contains

- Transformer semantic parser using Sentence-BERT embeddings.
- Canonical action/object/target ontology.
- Natural-language → TaskSpec metadata.
- Atomic action decomposition.
- Directed task/procedure graph.
- CV JSONL adapter.
- Stateful protocol verifier.
- Wrong object detection.
- Wrong target detection.
- Repeated-step detection.
- Out-of-order detection.
- Unknown/uncertain events.
- JSON metadata and verification outputs.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Parse the English procedure

```bash
python -m sih_nlp_v3.main parse --tasks tasks.txt
```

## 3. Build the graph

```bash
python -m sih_nlp_v3.main graph --tasks tasks.txt --out output
```

This creates:

- output/task_metadata.json
- output/task_graph.json

## 4. Connect CV

The CV teammate should emit JSONL events.

Example:

```json
{"timestamp":5.20,"action":"PLACE","object":"mouse","target":"slot 1","object_id":1,"person_id":1,"hand":"right","confidence":0.95}
```

Then:

```bash
python -m sih_nlp_v3.main verify \
    --tasks tasks.txt \
    --cv sample_cv_events.jsonl \
    --out output/verification.json
```

## Important integration contract

Your CV side should NOT send raw YOLO boxes directly to this module.

Prefer:

    YOLO + Pose + Hands + Tracking
                ↓
       interaction/action model
                ↓
       structured CV event
                ↓
          this verifier

Minimum event schema:

```json
{
  "timestamp": 12.43,
  "action": "GRASP",
  "object": "MOUSE",
  "target": null,
  "object_id": 7,
  "person_id": 1,
  "hand": "right",
  "confidence": 0.93
}
```

## Why Transformer + graph?

The Transformer handles language variability:

- "Pick up the mouse"
- "Grab the mouse"
- "Take the mouse"
- "Lift the mouse"

They can all map toward the same canonical intent.

The graph handles procedural structure:

    STEP 1 → STEP 2 → STEP 3 → STEP 4

and also expands a step such as:

    PLACE MOUSE IN SLOT 1

into:

    REACH → GRASP → LIFT → MOVE → PLACE → RELEASE

The verifier then compares observed CV events against this expected state.

## Offline deployment

The first run may download the Sentence-Transformer model. For the final standalone machine:

1. Download/cache the model while connected.
2. Copy the Hugging Face model cache or save the model into the project.
3. Point `ModelConfig.embedding_model` to the local model directory.
4. Run with network disabled.

## V3 design principle

NLP does NOT determine whether the human really performed the action.

NLP creates:

    expected semantics + expected protocol

CV creates:

    observed actions + objects + targets + confidence

The verifier compares:

    EXPECTED  vs  OBSERVED

This separation keeps the system modular and explainable.
