"""
stage3_build.py — SAIL Stage 3: L2 placement (route-style classification).
Classify every concept into the expert-approved L2 skeleton (self-consistency),
evaluate on seeds AND all concepts.

Run standalone:  python stage3_build.py --ontology X.owl --mayo --model gpt-5.5 \
                   --mayo-env ENV --runs 5 --seeds seed_concepts.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from collections import Counter
from ontology_io import load_ontology
import run_pipeline as RP
from sail_common import tqdm, evaluate_multilabel, APPROVED, BUILD_RESULT, L1_ASSIGN


def stage_build(args, onto):
    # --build-out lets several models write to different files
    out_path = getattr(args, "build_out", None) or "agent_build_result.json"
    print("=== STAGE build: classify concepts into the APPROVED L2 skeleton ===")
    print("    (route-style: same mechanism as L1 routing, target = L2 slots)")
    world = RP.L2World(onto)
    import numpy as np
    from collections import Counter

    # ---- L2 target slots = expert-approved skeleton (or GT L2 fallback) ----
    if os.path.exists(APPROVED):
        appr = json.load(open(APPROVED))
        approved_labels = {p["parent"] for p in appr.get("l2_parents", [])}
        # map each L2 label -> its L1 branch (from approved skeleton)
        l2_branch = {p["parent"]: p.get("l1", "") for p in appr.get("l2_parents", [])}
        slots = [c for c in world.L2 if world.lab(c) in approved_labels]
        extra = [world.o.lid(l) for l in approved_labels
                 if l in world.o.label2id and world.o.lid(l) not in set(world.L2)]
        L2 = sorted(set(slots) | set(extra), key=world.lab)
        print(f"  target = expert-approved skeleton: {len(L2)} L2 slots")
    else:
        L2 = world.L2
        l2_branch = {}
        print(f"  {APPROVED} not found; target = GT L2 slots ({len(L2)}) [ablation]")
    if not L2:
        raise SystemExit("no L2 target slots; run propose + approve first.")

    # mind-body concepts (route-derived phys∩psych): they are classified ONLY
    # into mind-body-branch L2 categories; in Physical/Psychological they stay
    # attached at the BRANCH level (not subdivided into L2). This realizes the
    # multi-parent rule: {mind-body L2, Physical branch, Psychological branch}.
    MB_LABEL = "Mind-body_Therapy"   # normalized (hyphen); match dash-insensitive
    mb_concept_labels = set()
    if os.path.exists(L1_ASSIGN):
        mb_concept_labels = set(json.load(open(L1_ASSIGN)).get(
            "mind_body_concepts", []))
    def _norm_dash(s):
        return s.replace("–", "-").replace("—", "-")
    mb_L2 = [s for s in L2 if _norm_dash(l2_branch.get(world.lab(s), "")) == MB_LABEL]
    phys_id = world.o.lid("Physical_Intervention")
    psych_id = world.o.lid("Psychological_Intervention")

    judge, _, monitor = RP.build_backend(args, world)

    # ---- which concepts to place: those under Physical/Psych/Mind-body per route,
    #      else all concepts (build the whole ontology) ----
    place_ids = world.concepts

    # ---- L2 classifier prompt (route-style; target = L2 categories) ----
    L2_labels = [world.lab(s) for s in L2]
    # give each L2 category a few example members so the model knows what the
    # category actually covers (category names alone are often opaque)
    def cat_hint(slot):
        kids = [world.lab(k) for k in onto.children.get(slot, [])][:3]
        return f"- {world.lab(slot)}" + (f" (e.g. {', '.join(kids)})" if kids else "")
    cat_block = "\n".join(cat_hint(s) for s in L2)

    BUILD_SYS = f"""You are an expert in complementary medicine ontology development. Your task is to assign a complementary-medicine concept to its parent category (or categories) among the L2 categories listed below.

Classify by the INTERVENTION ITSELF — how it is delivered or administered — NOT by its therapeutic effects, the diseases it treats, or its biological mechanisms.

L2 categories (with example members):
{cat_block}

Decision rules
1. Choose ONLY from the listed category labels; use the exact label spelling.
2. Assign multiple categories ONLY when the concept genuinely belongs to several (e.g. a technique that is intrinsically both a massage and an acupressure method).
3. If several categories could apply, pick the one that best represents the concept's primary modality.
4. Every concept must be assigned to at least one category.

Output format
Respond ONLY with a valid JSON object of this schema:
{{"parents": ["<category>"]}}
Constraints:
- "parents" must contain one or more labels.
- Each label must be exactly one of the listed L2 categories.
- No duplicate labels.
- No explanations.
- No reasoning.
- No confidence scores.
- No markdown.
- No text before or after the JSON object."""

    def place_one(cid, target):
        """Classify concept into `target` L2 slots (route-style)."""
        if args.mock:
            return judge.place(cid, target)
        node = onto.nodes[cid]
        target_labels = [world.lab(s) for s in target]
        sys = BUILD_SYS_MB if target is mb_L2 else BUILD_SYS
        u = [f"Concept: {onto.lab(cid)}"]
        if node.defn:
            u.append(f"Definition: {node.defn[:300]}")
        if node.alt:
            u.append(f"Also known as: {', '.join(node.alt[:5])}")
        u.append("Output:")
        r = (judge.c.choose(sys, "\n".join(u)) or {}
             ) if hasattr(judge, "c") else {}
        picked = set(r.get("parents", []) or [])
        return {s for s in target if world.lab(s) in picked}

    def place_multi(cid, runs, target):
        # keep model order identical for every concept when --dual is used
        _c = getattr(judge, "c", None)
        _inner = getattr(_c, "inner", _c)
        if hasattr(_inner, "reset_rotation"):
            _inner.reset_rotation()
        per = []
        for _ in range(runs):
            p = place_one(cid, target)
            per.append(p if p else set())
        cnt = Counter(frozenset(p) for p in per)
        best, freq = cnt.most_common(1)[0]
        return per, set(best), freq / runs

    # mind-body-specific prompt (target = mind-body L2 categories only)
    mb_cat_block = "\n".join(cat_hint(s) for s in mb_L2) if mb_L2 else ""
    BUILD_SYS_MB = BUILD_SYS.replace(cat_block, mb_cat_block) if mb_L2 else BUILD_SYS

    runs = max(1, args.runs)
    assign, confidence = {}, {}
    per_run_assign = [dict() for _ in range(runs)]   # each run's per-concept result
    for c in tqdm(place_ids, desc=f"build->L2 (x{runs})", unit="c"):
        if args.max_calls and monitor and monitor.ok + monitor.fail >= args.max_calls:
            break
        is_mb = onto.lab(c) in mb_concept_labels
        if is_mb and mb_L2:
            # mind-body concept: classify ONLY into mind-body-branch L2,
            # then attach at Physical & Psychological BRANCH level (rule).
            per, maj, conf = place_multi(c, runs, mb_L2)
            maj = set(maj) | {phys_id, psych_id}
            per = [set(p) | {phys_id, psych_id} for p in per]
        else:
            per, maj, conf = place_multi(c, runs, L2)
        assign[c] = maj
        confidence[c] = conf
        for i in range(min(runs, len(per))):
            per_run_assign[i][c] = per[i]

    # ---- evaluate on BOTH scopes: seeds and all concepts ----
    def gold_l2_set(c):
        return set(world.gold_l2(c)) & set(L2)

    placed_ids = [c for c in place_ids if c in assign and gold_l2_set(c)]
    seed_ids = []
    if args.seeds and os.path.exists(args.seeds):
        seed_labels = set(json.load(open(args.seeds))["seed_concepts"])
        seed_ids = [c for c in placed_ids if onto.lab(c) in seed_labels]

    def full_eval(eval_ids):
        if not eval_ids:
            return None
        Y_true = np.array([[1 if s in gold_l2_set(c) else 0 for s in L2]
                           for c in eval_ids])
        def metrics_of(assignment):
            Yp = np.array([[1 if s in assignment.get(c, set()) else 0
                            for s in L2] for c in eval_ids])
            return evaluate_multilabel(Y_true, Yp, L2_labels)
        # (A) per-run mean ± std (single-run stability vs voting)
        keys = ["subset_accuracy", "hamming_loss", "micro_f1", "macro_f1",
                "weighted_f1", "example_P", "example_R", "example_F1"]
        mean_std = {}
        if runs > 1 and per_run_assign:
            prm = [metrics_of(per_run_assign[i]) for i in range(len(per_run_assign))]
            for k in keys:
                vals = [mm[k] for mm in prm]
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
                sub = {c: assign.get(c, set()) for c in ids}
                Yt = np.array([[1 if s in gold_l2_set(c) else 0 for s in L2]
                               for c in ids])
                Yp = np.array([[1 if s in sub[c] else 0 for s in L2] for c in ids])
                mm = evaluate_multilabel(Yt, Yp, L2_labels)
                acc = sum(1 for c in ids if assign.get(c, set()) == gold_l2_set(c))
                return {"n": len(ids), "subset_acc": round(acc / len(ids), 3),
                        "micro_f1": mm["micro_f1"], "example_F1": mm["example_F1"]}
            m["confidence_analysis"] = {
                "threshold": args.conf_threshold,
                "high_conf": block(hi), "low_conf": block(lo),
                "expert_review_fraction": round(len(lo) / len(eval_ids), 3)}
        return m

    metrics_seed = full_eval(seed_ids) if seed_ids else None
    metrics_all = full_eval(placed_ids)
    metrics = metrics_seed or metrics_all or {}
    if metrics_seed:
        print(f"  evaluated on {len(seed_ids)} SEED concepts AND "
              f"all {len(placed_ids)} concepts with L2 gold")
    else:
        print(f"  evaluated on all {len(placed_ids)} concepts with L2 gold")

    # ---- report ----
    if metrics:
        print("\n=== L2 BUILD EVALUATION (multi-label, vs GT, on seeds) ===")
        print(f"  Subset Acc={metrics['subset_accuracy']:.3f}  "
              f"Hamming={metrics['hamming_loss']:.3f}")
        print(f"  Micro-F1={metrics['micro_f1']:.3f}  "
              f"Macro-F1={metrics['macro_f1']:.3f}  "
              f"Weighted-F1={metrics['weighted_f1']:.3f}")
        print(f"  Example-based P={metrics['example_P']:.3f} "
              f"R={metrics['example_R']:.3f} F1={metrics['example_F1']:.3f}")
        if "confidence_analysis" in metrics:
            ca = metrics["confidence_analysis"]
            hc, lc = ca.get("high_conf"), ca.get("low_conf")
            print(f"  confidence(thr {ca['threshold']}): ", end="")
            if hc:
                print(f"high n={hc['n']} acc={hc['subset_acc']} F1={hc['micro_f1']}, ",
                      end="")
            if lc:
                print(f"low n={lc['n']} acc={lc['subset_acc']} F1={lc['micro_f1']} <-review",
                      end="")
            print(f"  | expert-review {ca['expert_review_fraction']:.1%}")
        # single-run vs voting
        msd = metrics.get("per_run_mean_std", {})
        if msd.get("subset_accuracy"):
            sa = msd["subset_accuracy"]
            print(f"  single-run subset-acc={sa['mean']}±{sa['std']} "
                  f"vs voted={metrics['subset_accuracy']} (self-consistency gain)")
        if metrics_seed and metrics_all:
            print(f"\n  Scope comparison (seeds vs all):")
            print(f"      {'metric':<18}{'seeds(n=%d)' % metrics_seed['n_concepts']:>16}"
                  f"{'all(n=%d)' % metrics_all['n_concepts']:>16}")
            for k in ["subset_accuracy", "micro_f1", "macro_f1", "hamming_loss"]:
                print(f"      {k:<18}{metrics_seed[k]:>16.3f}{metrics_all[k]:>16.3f}")
    if monitor:
        print(f"\n  LLM calls: {monitor.ok} ok, {monitor.fail} failed")

    low_conf = sorted([{"concept": onto.lab(c), "confidence": confidence[c],
                        "l2": sorted(world.lab(s) for s in assign[c])}
                       for c in assign if confidence.get(c, 1.0) < args.conf_threshold],
                      key=lambda x: x["confidence"])
    json.dump({"runs": runs,
               "n_l2_slots": len(L2),
               "assignment": {world.lab(c): sorted(world.lab(s) for s in slots)
                              for c, slots in assign.items()},
               "confidence": {world.lab(c): confidence[c] for c in assign},
               "low_confidence_for_review": low_conf,
               "metrics_seeds": metrics_seed,
               "metrics_all": metrics_all},
              open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path} "
          f"({len(low_conf)} low-confidence flagged for review)")


# ===================================================================== #


def build_args():
    ap = argparse.ArgumentParser(description="SAIL Stage 3: L2 placement")
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
    stage_build(args, load_ontology(args.ontology))