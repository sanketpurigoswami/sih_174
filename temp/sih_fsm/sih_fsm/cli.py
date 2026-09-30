import argparse
import json
from .io import write_results_csv, write_results_json, write_results_jsonl
from .runner import run, summary


def main():
    p = argparse.ArgumentParser(description="SIH PS 26174 Protocol FSM")
    p.add_argument("--procedure", required=True, help="NLP output JSON (task metadata/procedure)")
    p.add_argument("--observed", required=True, help="CV observed events: JSONL or JSON")
    p.add_argument("--out", default="output", help="Output directory")
    p.add_argument("--min-confidence", type=float, default=0.55)
    args = p.parse_args()

    fsm, results = run(args.procedure, args.observed, args.min_confidence)
    write_results_json(results, f"{args.out}/verification.json")
    write_results_jsonl(results, f"{args.out}/verification.jsonl")
    write_results_csv(results, f"{args.out}/verification.csv")

    report = summary(fsm, results)
    with open(f"{args.out}/summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
