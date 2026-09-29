"""
merge_l1.py
===========
Produce the Stage-1 output that feeds Stage 2, from two model results:

  - concepts where BOTH models agree  -> accept that label set
  - concepts where they DISAGREE      -> resolved by "expert review"

Resolution modes for the disagreements (--resolve):
  gold      use the existing ontology's labels  (SIMULATED expert review —
            must be described as such in the paper; it makes downstream
            results optimistic because Stage-1 errors are corrected)
  primary   use model A's answer  (no expert; honest end-to-end baseline)
  union     union of both models   (high recall, e.g. recovers mind-body
            concepts one model missed)
  sheet     read decisions from a reviewed CSV (--review-csv), column
            'expert_decision', semicolon-separated labels

Usage:
  python merge_l1.py --ontology COMBO.owl \
      --a l1_gpt.json --b l1_qwen.json \
      --resolve gold --out l1_merged.json
"""
from __future__ import annotations
import argparse, csv, json
from ontology_io import load_ontology
import run_pipeline as RP
from sail_common import gold_l1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", required=True)
    ap.add_argument("--a", required=True, help="model A result (primary)")
    ap.add_argument("--b", required=True, help="model B result")
    ap.add_argument("--resolve", choices=["gold", "primary", "union", "sheet"],
                    default="gold")
    ap.add_argument("--review-csv", default=None,
                    help="(--resolve sheet) reviewed CSV with expert_decision")
    ap.add_argument("--out", default="l1_merged.json")
    args = ap.parse_args()

    onto = load_ontology(args.ontology)
    world = RP.L2World(onto)
    A = json.load(open(args.a))
    B = json.load(open(args.b))
    aa, bb = A.get("assignment", {}), B.get("assignment", {})

    sheet = {}
    if args.resolve == "sheet":
        if not args.review_csv:
            raise SystemExit("--resolve sheet requires --review-csv")
        with open(args.review_csv) as f:
            for row in csv.DictReader(f):
                dec = (row.get("expert_decision") or "").strip()
                if dec:
                    sheet[row["concept"]] = [s.strip() for s in dec.split(";")
                                             if s.strip()]

    merged, prov = {}, {}
    n_agree = n_resolved = n_unresolved = 0
    for c in aa:
        if c not in bb:
            merged[c] = aa[c]; prov[c] = "only_A"; continue
        sa, sb = set(aa[c]), set(bb[c])
        if sa == sb:
            merged[c] = sorted(sa); prov[c] = "agreed"; n_agree += 1
            continue
        # --- disagreement -> resolve ---
        if args.resolve == "gold":
            g = sorted(world.lab(s) for s in gold_l1(world, world.o.lid(c)))
            merged[c] = g or sorted(sa)
            prov[c] = "expert_simulated_gold"
        elif args.resolve == "primary":
            merged[c] = sorted(sa); prov[c] = "primary_A"
        elif args.resolve == "union":
            merged[c] = sorted(sa | sb); prov[c] = "union"
        else:                                   # sheet
            if c in sheet:
                merged[c] = sorted(sheet[c]); prov[c] = "expert_reviewed"
            else:
                merged[c] = sorted(sa); prov[c] = "unreviewed_fallback_A"
                n_unresolved += 1
        n_resolved += 1

    # mind-body: recompute from the merged L1 (Physical AND Psychological)
    def norm(s): return s.replace("\u2013", "-")
    mb = [c for c, labs in merged.items()
          if any(norm(l) == "Physical_Intervention" for l in labs)
          and any(norm(l) == "Psychological_Intervention" for l in labs)]

    out = {"runs": 1,
           "assignment": merged,
           "provenance": prov,
           "mind_body_concepts": sorted(mb),
           "merge_info": {"source_a": args.a, "source_b": args.b,
                          "resolve": args.resolve,
                          "n_agreed": n_agree,
                          "n_disagreed_resolved": n_resolved,
                          "n_unresolved": n_unresolved}}
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)

    total = len(merged)
    print(f"=== merged Stage-1 output ===")
    print(f"  concepts        : {total}")
    print(f"  agreed          : {n_agree}  ({n_agree/total:.1%})")
    print(f"  disagreed       : {n_resolved}  ({n_resolved/total:.1%})"
          f"  -> resolved by '{args.resolve}'")
    if n_unresolved:
        print(f"  UNREVIEWED      : {n_unresolved} (fell back to model A)")
    print(f"  mind-body concepts: {len(mb)}")
    if args.resolve == "gold":
        print("\n  NOTE: '--resolve gold' SIMULATES expert review using the "
              "existing ontology.\n        Describe it as simulated in the "
              "paper; downstream results are optimistic.")
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()