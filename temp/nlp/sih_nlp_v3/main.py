"""
CLI entry point.

Examples:

python -m sih_nlp_v3.main parse --tasks tasks.txt
python -m sih_nlp_v3.main graph --tasks tasks.txt
python -m sih_nlp_v3.main verify --tasks tasks.txt --cv cv_events.jsonl

For the first run, the Transformer model is downloaded from Hugging Face.
After that, keep the model cache available and the system can run offline.
"""

import argparse
import json
from pathlib import Path

from .pipeline import NLPV3Pipeline

def main():
    parser = argparse.ArgumentParser(description="SIH PS 26174 NLP V3")
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse")
    p_parse.add_argument("--tasks", required=True)

    p_graph = sub.add_parser("graph")
    p_graph.add_argument("--tasks", required=True)
    p_graph.add_argument("--out", default="output")

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--tasks", required=True)
    p_verify.add_argument("--cv", required=True)
    p_verify.add_argument("--out", default="output/verification.json")

    args = parser.parse_args()

    pipe = NLPV3Pipeline()

    if args.command == "parse":
        pipe.build_from_tasks(args.tasks)
        print(json.dumps([x.to_dict() for x in pipe.tasks], indent=2))

    elif args.command == "graph":
        pipe.build_from_tasks(args.tasks)
        pipe.export_metadata(args.out)
        print(f"Saved metadata and graph to: {Path(args.out).resolve()}")

    elif args.command == "verify":
        pipe.build_from_tasks(args.tasks)
        pipe.export_metadata(str(Path(args.out).parent))
        results = pipe.verify_cv_jsonl(args.cv, args.out)
        print(json.dumps(results, indent=2))
        print(f"\nSaved verification to: {Path(args.out).resolve()}")

if __name__ == "__main__":
    main()
