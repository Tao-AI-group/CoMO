"""
stage1_route.py — SAIL Stage 1: L1 routing.
Route each concept into the 4 NCCIH branches (multi-label, self-consistency),
derive mind-body (phys∩psych), evaluate on seeds AND all concepts.

Run standalone:  python stage1_route.py --ontology X.owl --mayo --model gpt-5.5 \
                   --mayo-env ENV --runs 5 --seeds seed_concepts.json
Or via:          python construction_agent.py --stage route ...
"""
from __future__ import annotations
import argparse, json, os
from ontology_io import load_ontology
import run_pipeline as RP
from sail_common import tqdm, evaluate_multilabel, gold_l1, L1_ASSIGN


def stage_route(args, onto):
    # output file: --route-out overrides the default, so running several
    # models does not overwrite each other's results
    out_path = getattr(args, "route_out", None) or L1_ASSIGN
    print("=== STAGE route: concepts -> NCCIH L1 (LLM), evaluated vs GT ===")
    world = RP.L2World(onto)
    judge, _, monitor = RP.build_backend(args, world)

    # L1 slots = the 4 NCCIH branches. Multi-label allowed. Rule:
    #   if a concept is routed to BOTH Physical AND Psychological, also attach
    #   Mind-body_Therapy (structure A: the concept is directly under all three).
    #   Other multi-label combos (e.g. Physical+Dietary) do NOT trigger mind-body.
    L1 = world.L1
    lid = {onto.lab(x): x for x in L1}
    PHYS = lid.get("Physical_Intervention")
    PSYCH = lid.get("Psychological_Intervention")

    # route-specific L1 classifier (multi-label; only the 4 NCCIH branches).
    # mind-body is NOT handled here — it is derived AFTER routing.
    ROUTE_SYS = """You are an expert in complementary medicine ontology development. Your task is to classify a complementary-medicine concept into one or more intervention branches.

Classify by the INTERVENTION ITSELF — how it is delivered or administered — NOT by its therapeutic effects, the diseases it treats, or its biological mechanisms.

Intervention branches
- Physical_Intervention: delivered through physical manipulation, body movement, manual techniques, touch, mechanical stimulation, or application of physical energy to the body.
- Psychological_Intervention: delivered through cognitive, emotional, behavioral, meditative, or other mental processes.
- Dietary_Intervention: intentionally modifies dietary intake, eating patterns, meal composition, nutritional behavior, or involves dietary supplementation for health purposes.
- Natural_Product_Intervention: primarily uses natural medicinal products (herbs, botanicals, essential oils, homeopathic preparations, natural therapeutic substances), excluding dietary supplementation.

Decision rules
1. Classify by the intervention itself, not by its effects, target diseases, or mechanisms.
2. Assign multiple branches ONLY when multiple modalities are intrinsic components of the intervention (not because it may produce physical/psychological/nutritional effects).
3. First decide all branches that genuinely apply and assign all of them.
4. If, after that, no branch clearly applies or it is genuinely ambiguous, assign the single branch that best represents the primary modality.

Examples (same input format as your task)
Concept: Tai Chi
Output: {"branches":["Physical_Intervention","Psychological_Intervention"]}

Concept: Mindfulness Meditation
Output: {"branches":["Psychological_Intervention"]}

Concept: Mediterranean Diet
Output: {"branches":["Dietary_Intervention"]}

Concept: Phytotherapy
Output: {"branches":["Natural_Product_Intervention"]}

Concept: Acupuncture
Output: {"branches":["Physical_Intervention"]}

Output format
Respond ONLY with a valid JSON object of this schema:
{"branches": ["<branch>"]}
Constraints:
- "branches" contains one or more labels.
- Each label must be exactly one of: Physical_Intervention, Psychological_Intervention, Dietary_Intervention, Natural_Product_Intervention.
- No duplicate labels. No explanations, reasoning, confidence scores, or markdown. No text before or after the JSON object."""

    def route_one(cid):
        if args.mock:
            return judge.place(cid, L1)          # mock uses GT-aware place
        # User prompt = this concept's data (name + definition + aliases),
        # same format as the few-shot examples above.
        node = onto.nodes[cid]
        u = [f"Concept: {onto.lab(cid)}"]
        if node.defn:
            u.append(f"Definition: {node.defn[:300]}")
        if node.alt:
            u.append(f"Also known as: {', '.join(node.alt[:5])}")
        u.append("Output:")
        r = (judge.c.choose(ROUTE_SYS, "\n".join(u)) or {}
             ) if hasattr(judge, "c") else {}
        picked = set(r.get("branches", []) or [])
        return {s for s in L1 if onto.lab(s) in picked}

    def route_multi(cid, runs):
        """Route `runs` times. Return (per_run_list, majority_set, confidence)."""
        from collections import Counter
        # keep model order identical for every concept when --dual is used
        _c = getattr(judge, "c", None)
        _inner = getattr(_c, "inner", _c)
        if hasattr(_inner, "reset_rotation"):
            _inner.reset_rotation()
        per_run = []
        for _ in range(runs):
            pred = route_one(cid)
            per_run.append(pred if pred else {L1[0]})
        cnt = Counter(frozenset(p) for p in per_run)
        best, freq = cnt.most_common(1)[0]
        return per_run, set(best), freq / runs

    runs = max(1, args.runs)
    assign, confidence = {}, {}           # assign = majority-vote result
    per_run_assign = [dict() for _ in range(runs)]   # each run's raw result
    for c in tqdm(world.concepts, desc=f"route->L1 (x{runs})", unit="c"):
        if args.max_calls and monitor and monitor.ok + monitor.fail >= args.max_calls:
            break
        runs_out, maj, conf = route_multi(c, runs)
        assign[c] = maj
        confidence[c] = conf
        for i, p in enumerate(runs_out):
            per_run_assign[i][c] = p

    # ---- evaluate on BOTH scopes: seeds (Scheme Y) and all concepts ----
    import numpy as np
    labels = L1
    lab_names = [onto.lab(s) for s in labels]
    routed_ids = [c for c in world.concepts if c in assign]

    seed_ids = []
    if args.seeds and os.path.exists(args.seeds):
        seed_labels = set(json.load(open(args.seeds))["seed_concepts"])
        seed_ids = [c for c in routed_ids if onto.lab(c) in seed_labels]

    def full_eval(eval_ids):
        """One complete metric block for a given concept scope."""
        if not eval_ids:
            return None
        Y_true = np.array([[1 if s in gold_l1(world, c) else 0 for s in labels]
                           for c in eval_ids])
        def metrics_of(assignment):
            Yp = np.array([[1 if s in assignment.get(c, set()) else 0
                            for s in labels] for c in eval_ids])
            return evaluate_multilabel(Y_true, Yp, lab_names)
        # (A) per-run mean ± std
        keys = ["subset_accuracy", "hamming_loss", "micro_f1", "macro_f1",
                "weighted_f1", "example_P", "example_R", "example_F1"]
        mean_std = {}
        if runs > 1:
            prm = [metrics_of(per_run_assign[i]) for i in range(runs)]
            for k in keys:
                vals = [m[k] for m in prm]
                mean_std[k] = {"mean": round(float(np.mean(vals)), 3),
                               "std": round(float(np.std(vals)), 3)}
        # (B) majority-vote metrics
        m = metrics_of(assign)
        m["per_run_mean_std"] = mean_std
        m["n_concepts"] = len(eval_ids)
        # (C) confidence analysis: subset-acc AND F1 for high vs low conf
        if runs > 1:
            hi = [c for c in eval_ids if confidence[c] >= args.conf_threshold]
            lo = [c for c in eval_ids if confidence[c] < args.conf_threshold]
            def block(ids):
                if not ids:
                    return None
                Yt = np.array([[1 if s in gold_l1(world, c) else 0 for s in labels]
                               for c in ids])
                Yp = np.array([[1 if s in assign.get(c, set()) else 0
                                for s in labels] for c in ids])
                mm = evaluate_multilabel(Yt, Yp, lab_names)
                acc = sum(1 for c in ids
                          if assign.get(c, set()) == gold_l1(world, c))
                return {"n": len(ids), "subset_acc": round(acc / len(ids), 3),
                        "micro_f1": mm["micro_f1"], "example_F1": mm["example_F1"]}
            m["confidence_analysis"] = {
                "threshold": args.conf_threshold,
                "high_conf": block(hi), "low_conf": block(lo),
                "expert_review_fraction": round(len(lo) / len(eval_ids), 3)}
        # (D) mind-body P/R on this scope
        p = [c for c in eval_ids if PHYS in assign.get(c, set())
             and PSYCH in assign.get(c, set())]
        g = [c for c in eval_ids if PHYS in gold_l1(world, c)
             and PSYCH in gold_l1(world, c)]
        hit = len(set(p) & set(g))
        mp = hit / len(p) if p else 0
        mr = hit / len(g) if g else 0
        m["mindbody"] = {"P": round(mp, 3), "R": round(mr, 3),
                         "F1": round(2*mp*mr/(mp+mr), 3) if (mp+mr) else 0,
                         "pred_count": len(p), "gt_count": len(g)}
        return m

    metrics_seed = full_eval(seed_ids) if seed_ids else None
    metrics_all = full_eval(routed_ids)
    if metrics_seed:
        print(f"  evaluated on {len(seed_ids)} SEED concepts (from {args.seeds}) "
              f"AND on all {len(routed_ids)} concepts")
    else:
        print(f"  evaluated on all {len(routed_ids)} concepts "
              "(no --seeds given)")

    # primary metrics = seeds if available, else all
    metrics = metrics_seed or metrics_all

    # ---- DERIVE mind-body concepts from ALL routed concepts (for building) ----
    mb_all = [c for c in assign
              if PHYS in assign.get(c, set()) and PSYCH in assign.get(c, set())]
    mb_P = metrics["mindbody"]["P"]
    mb_R = metrics["mindbody"]["R"]
    mb_F = metrics["mindbody"]["F1"]
    mb_concepts = [c for c in (seed_ids or routed_ids)
                   if PHYS in assign.get(c, set()) and PSYCH in assign.get(c, set())]
    mb_gold = [c for c in (seed_ids or routed_ids)
               if PHYS in gold_l1(world, c) and PSYCH in gold_l1(world, c)]
    mean_std = metrics.get("per_run_mean_std", {})

    print("\n=== L1 ROUTE EVALUATION (multi-label, 4 NCCIH classes, vs GT) ===")
    if mean_std:
        print(f"\n  [A] Per-run stability ({runs} runs, mean ± std):")
        print(f"      Subset Acc = {mean_std['subset_accuracy']['mean']:.3f} "
              f"± {mean_std['subset_accuracy']['std']:.3f}")
        print(f"      Hamming    = {mean_std['hamming_loss']['mean']:.3f} "
              f"± {mean_std['hamming_loss']['std']:.3f}")
        print(f"      Micro-F1   = {mean_std['micro_f1']['mean']:.3f} "
              f"± {mean_std['micro_f1']['std']:.3f}")
        print(f"      Macro-F1   = {mean_std['macro_f1']['mean']:.3f} "
              f"± {mean_std['macro_f1']['std']:.3f}")
    print(f"\n  [B] Majority-vote result ({runs} runs -> self-consistency):")
    print(f"      1. Subset Accuracy = {metrics['subset_accuracy']:.3f}")
    print(f"      2. Hamming Loss    = {metrics['hamming_loss']:.3f}")
    print(f"      3. Micro-F1={metrics['micro_f1']:.3f}  "
          f"Macro-F1={metrics['macro_f1']:.3f}  "
          f"Weighted-F1={metrics['weighted_f1']:.3f}")
    print(f"      4. Example-based P={metrics['example_P']:.3f} "
          f"R={metrics['example_R']:.3f} F1={metrics['example_F1']:.3f}")
    print(f"      5. Per-label:")
    for lab, prf in metrics["per_label"].items():
        print(f"         {lab:26s} P={prf['P']:.3f} R={prf['R']:.3f} F1={prf['F1']:.3f}")
    print(f"      6. mind-body P={mb_P:.3f} R={mb_R:.3f} F1={mb_F:.3f} "
          f"(pred {len(mb_concepts)}/GT {len(mb_gold)})")
    if runs > 1 and "confidence_analysis" in metrics:
        ca = metrics["confidence_analysis"]
        hc, lc = ca.get("high_conf"), ca.get("low_conf")
        print(f"\n  [C] Confidence (threshold {ca['threshold']}):")
        if hc:
            print(f"      high-conf: n={hc['n']}  subset-acc={hc['subset_acc']}  "
                  f"micro-F1={hc['micro_f1']}")
        if lc:
            print(f"      low-conf:  n={lc['n']}  subset-acc={lc['subset_acc']}  "
                  f"micro-F1={lc['micro_f1']}   <- expert review")
        print(f"      expert-review fraction = {ca['expert_review_fraction']:.1%} "
              "of concepts (only low-conf need expert)")
    # ---- side-by-side: seeds vs all concepts ----
    if metrics_seed and metrics_all:
        print(f"\n  [D] Scope comparison (seeds vs all concepts):")
        print(f"      {'metric':<18}{'seeds(n=%d)' % metrics_seed['n_concepts']:>16}"
              f"{'all(n=%d)' % metrics_all['n_concepts']:>16}")
        for k in ["subset_accuracy", "micro_f1", "macro_f1", "hamming_loss"]:
            print(f"      {k:<18}{metrics_seed[k]:>16.3f}{metrics_all[k]:>16.3f}")
        print(f"      {'mind-body F1':<18}"
              f"{metrics_seed['mindbody']['F1']:>16.3f}"
              f"{metrics_all['mindbody']['F1']:>16.3f}")
    if monitor:
        print(f"\n  LLM calls: {monitor.ok} ok, {monitor.fail} failed")

    # low-confidence concepts WITH their confidence (for expert review targeting)
    low_conf_all = sorted(
        [{"concept": onto.lab(c), "confidence": confidence[c],
          "route": sorted(onto.lab(s) for s in assign[c])}
         for c in assign if confidence.get(c, 1.0) < args.conf_threshold],
        key=lambda x: x["confidence"])          # most uncertain first
    # full per-run raw results — so you can re-threshold / re-analyze WITHOUT
    # re-running the LLM (saves the 5x cost).
    per_run_serial = [{world.lab(c): sorted(onto.lab(s) for s in per_run_assign[i][c])
                       for c in per_run_assign[i]} for i in range(runs)]
    json.dump({"runs": runs,
               "assignment": {world.lab(c): sorted(world.lab(s) for s in slots)
                              for c, slots in assign.items()},
               "confidence": {world.lab(c): confidence[c] for c in assign},
               "per_run_assignments": per_run_serial,      # ← all runs kept
               "low_confidence_for_review": low_conf_all,  # ← with confidence
               "mind_body_concepts": [world.lab(c) for c in mb_all],
               "metrics_seeds": metrics_seed,   # ← Scheme Y primary (expert-annotated)
               "metrics_all": metrics_all},     # ← full-scope reference
              open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path}")
    print(f"  ({len(low_conf_all)} low-confidence concepts flagged, with "
          "confidence; all 5 runs saved for re-analysis without re-running)")
    print("Next: python construction_agent.py --stage propose ...")


# ===================================================================== #
#  STAGE: propose  (S1-S2, then pause)                                   #
# ===================================================================== #


def build_args():
    ap = argparse.ArgumentParser(description="SAIL Stage 1: L1 routing")
    ap.add_argument("--ontology", default="ontology.json")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--conf-threshold", type=float, default=0.6)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--mayo", action="store_true")
    ap.add_argument("--mayo-env", default=None)
    ap.add_argument("--max-calls", type=int, default=0)
    ap.add_argument("--fail-fast", type=int, default=5)
    ap.add_argument("--quiet-calls", action="store_true")
    ap.add_argument("--voters", type=int, default=1)
    ap.add_argument("--tau-vote", type=float, default=0.6)
    ap.add_argument("--use-defs", action="store_true")
    return ap


if __name__ == "__main__":
    args = build_args().parse_args()
    stage_route(args, load_ontology(args.ontology))