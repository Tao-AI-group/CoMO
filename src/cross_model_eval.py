"""
cross_model_eval.py
===================
Compare TWO independently-produced result files (one per model) and report:

  - each model's own accuracy (vs gold)
  - cross-model agreement rate
  - accuracy of the AGREED subset      -> what you can accept automatically
  - accuracy of the DISAGREED subset   -> what goes to expert review
  - expert-review fraction             -> = disagreement rate

This replaces "run both models together"; you run each model separately, save
its result JSON, then compare offline. No LLM calls here.

Usage (route):
  python cross_model_eval.py --ontology COMBO.owl --stage route \
      --a l1_gpt.json --b l1_qwen.json \
      --name-a gpt-5.5 --name-b qwen3 \
      --seeds seed_concepts_150.json

Usage (build):
  python cross_model_eval.py --ontology COMBO.owl --stage build \
      --a build_gpt.json --b build_qwen.json \
      --skeleton approved_skeleton.json \
      --seeds seed_concepts_150.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from ontology_io import load_ontology
import run_pipeline as RP
from sail_common import evaluate_multilabel, gold_l1


def load_assign(path):
    d = json.load(open(path))
    return d.get("assignment", {}), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", required=True)
    ap.add_argument("--stage", choices=["route", "build"], required=True)
    ap.add_argument("--a", required=True, help="model A result json")
    ap.add_argument("--b", required=True, help="model B result json")
    ap.add_argument("--name-a", default="modelA")
    ap.add_argument("--name-b", default="modelB")
    ap.add_argument("--skeleton", default="approved_skeleton.json",
                    help="(build) approved skeleton defining the L2 space")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--out", default="cross_model_eval.json")
    ap.add_argument("--review-csv", default=None,
                    help="export the disagreements as a CSV review sheet")
    ap.add_argument("--print-disagreements", type=int, default=0,
                    help="print the first N disagreements to stdout")
    args = ap.parse_args()

    onto = load_ontology(args.ontology)
    world = RP.L2World(onto)
    A, rawA = load_assign(args.a)
    B, rawB = load_assign(args.b)

    # ---------------- label space + gold ----------------------------- #
    if args.stage == "route":
        lab_names = [world.lab(world.o.lid(l)) for l in
                     ["Physical_Intervention", "Psychological_Intervention",
                      "Dietary_Intervention", "Natural_Product_Intervention"]
                     if l in world.o.label2id]
        def gold_of(clabel):
            cid = world.o.lid(clabel)
            return {world.lab(s) for s in gold_l1(world, cid)}
        drop_uncovered = False
    else:
        skel = set()
        if os.path.exists(args.skeleton):
            sk = json.load(open(args.skeleton))
            skel = {p["parent"] for p in sk.get("l2_parents", [])}
        if not skel:
            skel = {p for v in list(A.values()) + list(B.values()) for p in v}
            print("WARNING: no skeleton file; using union of predictions.")
        lab_names = sorted(skel)
        def gold_of(clabel):
            cid = world.o.lid(clabel)
            return {world.lab(s) for s in world.gold_l2(cid)} & skel
        drop_uncovered = True

    name2idx = {n: i for i, n in enumerate(lab_names)}
    def vec(labelset):
        v = [0] * len(lab_names)
        for l in labelset:
            if l in name2idx:
                v[name2idx[l]] = 1
        return v

    # ---------------- concept scope ---------------------------------- #
    common = [c for c in A if c in B]
    if drop_uncovered:
        common = [c for c in common if gold_of(c)]
    print(f"concepts in both files: {len(common)}"
          f"  ({args.name_a}:{len(A)}  {args.name_b}:{len(B)})")

    seed_labels = None
    if args.seeds:
        seed_labels = set(json.load(open(args.seeds)).get("seed_concepts", []))

    # ---------------- evaluation ------------------------------------- #
    def block(ids, pred_map, tag):
        if not ids:
            return None
        Yt = np.array([vec(gold_of(c)) for c in ids])
        Yp = np.array([vec(set(pred_map[c])) for c in ids])
        m = evaluate_multilabel(Yt, Yp, lab_names)
        # subset accuracy over the label space only: predictions may carry
        # labels outside it (e.g. branch-level labels added by the mind-body
        # multi-parent rule), which must not count as errors. Comparing the
        # vectors keeps this consistent with the per-stage evaluation.
        acc = sum(1 for i in range(len(ids))
                  if (Yt[i] == Yp[i]).all()) / len(ids)
        return {"n": len(ids), "subset_acc": round(float(acc), 3),
                "micro_f1": m["micro_f1"], "macro_f1": m["macro_f1"],
                "example_F1": m["example_F1"], "tag": tag}

    def evaluate(ids, scope_name):
        # agreement is judged within the label space, for the same reason as
        # subset accuracy above
        space = set(lab_names)
        def inside(c, mp):
            return set(mp[c]) & space
        agree = [c for c in ids if inside(c, A) == inside(c, B)]
        disagree = [c for c in ids if inside(c, A) != inside(c, B)]

        res = {
            "n_concepts": len(ids),
            f"{args.name_a}_alone": block(ids, A, args.name_a),
            f"{args.name_b}_alone": block(ids, B, args.name_b),
            "agreed": block(agree, A, "agreed"),
            "disagreed_A": block(disagree, A, f"disagreed({args.name_a})"),
            "disagreed_B": block(disagree, B, f"disagreed({args.name_b})"),
            "agreement_rate": round(len(agree) / len(ids), 3) if ids else 0,
            "expert_review_fraction": round(len(disagree) / len(ids), 3) if ids else 0,
        }
        # detailed list of disagreements, for the expert review sheet
        res["disagreement_list"] = [
            {"concept": c,
             args.name_a: sorted(inside(c, A)),
             args.name_b: sorted(inside(c, B)),
             "gold": sorted(gold_of(c)),
             "a_correct": inside(c, A) == gold_of(c),
             "b_correct": inside(c, B) == gold_of(c),
             "jaccard": round(len(inside(c, A) & inside(c, B)) /
                              len(inside(c, A) | inside(c, B)), 2)
                        if (inside(c, A) | inside(c, B)) else 0.0}
            for c in sorted(disagree)]

        # jaccard of the two predictions (partial agreement signal)
        js = []
        for c in ids:
            sa, sb = inside(c, A), inside(c, B)
            js.append(len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0)
        res["mean_jaccard"] = round(float(np.mean(js)), 3) if js else 0

        print(f"\n=== {scope_name} (n={len(ids)}) ===")
        for k in [f"{args.name_a}_alone", f"{args.name_b}_alone"]:
            r = res[k]
            if r:
                print(f"  {k:22s} subset-acc={r['subset_acc']}  "
                      f"micro-F1={r['micro_f1']}  macro-F1={r['macro_f1']}")
        print(f"  agreement rate      = {res['agreement_rate']:.1%}  "
              f"(mean Jaccard {res['mean_jaccard']})")
        ag, dA = res["agreed"], res["disagreed_A"]
        if ag:
            print(f"  AGREED   n={ag['n']:4d}  subset-acc={ag['subset_acc']}  "
                  f"<- auto-accept")
        if dA:
            print(f"  DISAGREED n={dA['n']:4d}  subset-acc={dA['subset_acc']} "
                  f"({args.name_a}) / {res['disagreed_B']['subset_acc']} "
                  f"({args.name_b})  <- expert review")
        print(f"  expert-review fraction = {res['expert_review_fraction']:.1%}")
        if ag and dA:
            gap = round(ag["subset_acc"] - dA["subset_acc"], 3)
            print(f"  discrimination (agreed - disagreed) = {gap}  "
                  f"{'GOOD' if gap > 0.15 else 'WEAK'}")
        if args.print_disagreements:
            print(f"\n  --- disagreements (first "
                  f"{min(args.print_disagreements, len(disagree))}) ---")
            for r in res["disagreement_list"][:args.print_disagreements]:
                mark = ("A✓" if r["a_correct"] else
                        "B✓" if r["b_correct"] else "both✗")
                print(f"    {r['concept'][:38]:40s} "
                      f"{args.name_a}={','.join(r[args.name_a])[:28]:30s} "
                      f"{args.name_b}={','.join(r[args.name_b])[:28]:30s} [{mark}]")
        return res

    out = {}
    if seed_labels:
        out["seeds"] = evaluate([c for c in common if c in seed_labels], "SEEDS")
    out["all"] = evaluate(common, "ALL")
    out["models"] = {"A": args.name_a, "B": args.name_b,
                     "file_a": args.a, "file_b": args.b}

    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"\nwrote {args.out}")

    # ---- expert review sheet (the disagreements) --------------------- #
    if args.review_csv:
        import csv
        scope = out.get("all") or out.get("seeds")
        rows = scope.get("disagreement_list", [])
        with open(args.review_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["concept", args.name_a, args.name_b, "jaccard",
                        "gold(reference)", "expert_decision"])
            for r in rows:
                w.writerow([r["concept"],
                            "; ".join(r[args.name_a]),
                            "; ".join(r[args.name_b]),
                            r["jaccard"],
                            "; ".join(r["gold"]),
                            ""])          # blank column for the expert
        print(f"wrote {args.review_csv}  ({len(rows)} concepts to review)")


if __name__ == "__main__":
    main()