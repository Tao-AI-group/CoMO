"""
recompute_metrics.py
====================
Recompute the new metrics (single-run mean±std, high/low-confidence F1,
expert-review fraction) from an ALREADY-SAVED route/build result — WITHOUT
re-calling the LLM. Uses the stored per_run_assignments + confidence + assignment.

Usage:
  # route result
  python recompute_metrics.py --ontology COMBO.owl \
      --result l1_assignment.json --stage route \
      --seeds seed_concepts.json --conf-threshold 0.6

  # build result
  python recompute_metrics.py --ontology COMBO.owl \
      --result agent_build_result.json --stage build \
      --seeds seed_concepts.json --conf-threshold 0.6
"""
from __future__ import annotations
import argparse, json
import numpy as np
from ontology_io import load_ontology
import run_pipeline as RP
from sail_common import evaluate_multilabel, gold_l1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", default="ontology.json")
    ap.add_argument("--result", required=True, help="saved l1_assignment.json / agent_build_result.json")
    ap.add_argument("--stage", choices=["route", "build"], required=True)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--conf-threshold", type=float, default=0.6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    onto = load_ontology(args.ontology)
    world = RP.L2World(onto)
    data = json.load(open(args.result))

    assign = data.get("assignment", {})              # label -> [parent labels]
    confidence = data.get("confidence", {})
    per_run = data.get("per_run_assignments", [])    # list of {label:[labels]}
    runs = len(per_run) if per_run else data.get("runs", 1)

    if not per_run:
        print("WARNING: no per_run_assignments in file; cannot compute "
              "single-run mean±std. (confidence F1 + expert fraction still ok.)")

    # target label space
    l2_branch = {}
    if args.stage == "route":
        labels = [world.o.lid(l) for l in
                  ["Physical_Intervention", "Psychological_Intervention",
                   "Dietary_Intervention", "Natural_Product_Intervention"]
                  if l in world.o.label2id]
        lab_names = [world.lab(s) for s in labels]
        def gold_of(clabel):
            cid = world.o.lid(clabel)
            return {world.lab(s) for s in gold_l1(world, cid)}
    else:
        # build: L2 space = the approved skeleton (same as build stage),
        # and gold is intersected with that skeleton (so skeleton omissions
        # are NOT counted as build errors) — matches stage3_build.gold_l2_set.
        import os
        skel_labels = set()
        if os.path.exists("approved_skeleton.json"):
            appr = json.load(open("approved_skeleton.json"))
            skel_labels = {p["parent"] for p in appr.get("l2_parents", [])}
        if skel_labels:
            lab_names = sorted(skel_labels)
        else:
            lab_names = sorted({p for v in assign.values() for p in v})
            print("WARNING: approved_skeleton.json not found; using assignment "
                  "label universe (may differ from build's L2 target).")
        skel_set = set(lab_names)
        def gold_of(clabel):
            cid = world.o.lid(clabel)
            return {world.lab(s) for s in world.gold_l2(cid)} & skel_set
        # L2 -> branch map (for per-branch per-label grouping, scheme B)
        l2_branch = {}
        if os.path.exists("approved_skeleton.json"):
            appr2 = json.load(open("approved_skeleton.json"))
            l2_branch = {p["parent"]: p.get("l1", "") for p in appr2.get("l2_parents", [])}

    # concept scope
    all_concepts = list(assign.keys())
    seed_labels = None
    if args.seeds:
        seed_labels = set(json.load(open(args.seeds)).get("seed_concepts", []))

    def eval_scope(concept_labels, drop_uncovered=True):
        # match build: by default drop concepts whose gold L2 is NOT in the
        # skeleton (gold_of empty) — skeleton omissions aren't build's fault.
        ids = [c for c in concept_labels if c in assign]
        if args.stage == "build" and drop_uncovered:
            ids = [c for c in ids if gold_of(c)]
        if not ids:
            return None
        name2idx = {n: i for i, n in enumerate(lab_names)}
        def to_vec(labelset):
            v = [0] * len(lab_names)
            for l in labelset:
                if l in name2idx:
                    v[name2idx[l]] = 1
            return v
        Y_true = np.array([to_vec(gold_of(c)) for c in ids])
        Y_pred = np.array([to_vec(set(assign[c])) for c in ids])
        m = evaluate_multilabel(Y_true, Y_pred, lab_names)
        m["n_concepts"] = len(ids)

        # (Scheme B) per-branch aggregation of per-label F1: group each L2 by its
        # branch (Physical / Psychological / Mind-body), average P/R/F1 within.
        if l2_branch and m.get("per_label"):
            import numpy as _np
            by_branch = {}
            for lab, prf in m["per_label"].items():
                br = l2_branch.get(lab, "Other")
                by_branch.setdefault(br, []).append(prf)
            m["per_branch_label"] = {
                br: {"n_l2": len(v),
                     "macro_P": round(float(_np.mean([x["P"] for x in v])), 3),
                     "macro_R": round(float(_np.mean([x["R"] for x in v])), 3),
                     "macro_F1": round(float(_np.mean([x["F1"] for x in v])), 3)}
                for br, v in by_branch.items()}

        # (A) single-run mean ± std
        if per_run and runs > 1:
            keys = ["subset_accuracy", "hamming_loss", "micro_f1", "macro_f1",
                    "weighted_f1", "example_P", "example_R", "example_F1"]
            prm = []
            for r in per_run:
                Yp = np.array([to_vec(set(r.get(c, []))) for c in ids])
                prm.append(evaluate_multilabel(Y_true, Yp, lab_names))
            m["per_run_mean_std"] = {
                k: {"mean": round(float(np.mean([x[k] for x in prm])), 3),
                    "std": round(float(np.std([x[k] for x in prm])), 3)}
                for k in keys}

        # (B) confidence analysis: subset-acc + F1, high vs low
        hi = [c for c in ids if confidence.get(c, 1.0) >= args.conf_threshold]
        lo = [c for c in ids if confidence.get(c, 1.0) < args.conf_threshold]
        def block(cids):
            if not cids:
                return None
            Yt = np.array([to_vec(gold_of(c)) for c in cids])
            Yp = np.array([to_vec(set(assign[c])) for c in cids])
            mm = evaluate_multilabel(Yt, Yp, lab_names)
            acc = sum(1 for c in cids if set(assign[c]) == gold_of(c))
            return {"n": len(cids), "subset_acc": round(acc / len(cids), 3),
                    "micro_f1": mm["micro_f1"], "example_F1": mm["example_F1"]}
        m["confidence_analysis"] = {
            "threshold": args.conf_threshold,
            "high_conf": block(hi), "low_conf": block(lo),
            "expert_review_fraction": round(len(lo) / len(ids), 3)}
        return m

    result = {}
    if seed_labels:
        result["metrics_seeds"] = eval_scope(seed_labels)
    result["metrics_all"] = eval_scope(all_concepts)

    # print
    for scope in ["metrics_seeds", "metrics_all"]:
        m = result.get(scope)
        if not m:
            continue
        print(f"\n=== {scope} (n={m['n_concepts']}) ===")
        print(f"  Subset-Acc={m['subset_accuracy']}  Micro-F1={m['micro_f1']}  "
              f"Macro-F1={m['macro_f1']}  Weighted-F1={m['weighted_f1']}  "
              f"Example-F1={m['example_F1']}")
        msd = m.get("per_run_mean_std", {})
        if msd.get("subset_accuracy"):
            sa = msd["subset_accuracy"]
            print(f"  single-run subset-acc={sa['mean']}±{sa['std']} "
                  f"vs voted={m['subset_accuracy']} (self-consistency gain)")
        ca = m["confidence_analysis"]
        hc, lc = ca.get("high_conf"), ca.get("low_conf")
        if hc:
            print(f"  high-conf: n={hc['n']} acc={hc['subset_acc']} F1={hc['micro_f1']}")
        if lc:
            print(f"  low-conf:  n={lc['n']} acc={lc['subset_acc']} F1={lc['micro_f1']} <-review")
        print(f"  expert-review fraction = {ca['expert_review_fraction']:.1%}")
        # (Scheme B) per-branch macro-F1
        pb = m.get("per_branch_label")
        if pb:
            print("  per-branch macro (scheme B):")
            for br, s in pb.items():
                print(f"    {br:28s} n_L2={s['n_l2']:2d}  "
                      f"macro-P={s['macro_P']}  macro-R={s['macro_R']}  "
                      f"macro-F1={s['macro_F1']}")
        # (Scheme A) worst L2 categories dragging macro-F1 down
        pl = m.get("per_label")
        if pl:
            zeros = [l for l, prf in pl.items() if prf["F1"] == 0.0]
            print(f"  L2 categories with F1=0: {len(zeros)}/{len(pl)} "
                  f"(these drag macro-F1 down)")
            if zeros:
                print(f"    e.g. {', '.join(zeros[:8])}")

    out = args.out or args.result.replace(".json", "_recomputed.json")
    json.dump(result, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()