"""
stage2_propose.py — SAIL Stage 2: L2 skeleton proposal (+ emit-approved).
Cluster each branch + LLM proposes L2 parents (from concepts). Pauses for expert.

Run standalone:  python stage2_propose.py --stage propose --ontology X.owl \
                   --branches "..." --embed st --cluster ap --propose llm ...
                 python stage2_propose.py --stage emit-approved --ontology X.owl
"""
from __future__ import annotations
import argparse, json, os
from ontology_io import load_ontology
import run_pipeline as RP
import cluster_map as CM
from sail_common import tqdm, L1_ASSIGN, PROPOSED, APPROVED


def stage_propose(args, onto):
    print("=== STAGE propose: anchor + skeleton proposal (then PAUSE) ===")
    # client for LLM parent proposal.
    # llm-direct REQUIRES a client; --propose lexical with --propose-mode
    # llm-direct would silently fall back to a lexical heuristic (no LLM call),
    # so treat llm-direct as implying --propose llm.
    needs_llm = (args.propose == "llm" or
                 getattr(args, "propose_mode", "") == "llm-direct")
    if needs_llm and args.propose != "llm":
        print("  note: --propose-mode llm-direct implies --propose llm; "
              "enabling the LLM client.")
    client = None
    if needs_llm:
        if args.mock:
            from build_eval import MockClient; client = MockClient()
        elif args.mayo:
            from mayo_client import MayoClient
            client = MayoClient(engine=args.model, env_path=args.mayo_env)
        else:
            from build_eval import VLLMClient
            client = VLLMClient(args.model, args.api_base, args.api_key)

    # mind-body excluded from Physical/Psychological, gets its own plot
    MB = "Mind–body_Therapy"
    mb_excl = (onto.subtree(MB) | {onto.lid(MB)}) if MB in onto.label2id else set()
    split_branches = {"Physical_Intervention", "Psychological_Intervention"}

    branches = args.branches.split(",")

    # Determine concept membership per L1 branch:
    #  - default: use the LLM route result (l1_assignment.json) — no GT.
    #  - --use-gt-l1: use GT subtree (for comparison / ablation).
    branch_concepts = {}
    if not args.use_gt_l1 and os.path.exists(L1_ASSIGN):
        world = RP.L2World(onto)
        l1data = json.load(open(L1_ASSIGN))
        routed = l1data["assignment"]
        mb_derived = set(l1data.get("mind_body_concepts", []))  # phys∩psych
        lab2id = {world.lab(c): c for c in world.concepts}
        for b in branches:
            members = set()
            # normalize dashes so "Mind-body" (hyphen) and "Mind–body" (en-dash)
            # both match the mind-body branch
            b_norm = b.replace("–", "-").replace("—", "-")
            if b_norm == "Mind-body_Therapy":
                # mind-body branch = concepts routed to BOTH phys & psych
                members = {lab2id[l] for l in mb_derived if l in lab2id}
            else:
                for clab, l1labs in routed.items():
                    if b in l1labs and clab in lab2id:
                        members.add(lab2id[clab])
            branch_concepts[b] = members
        print(f"  using LLM route result from {L1_ASSIGN} (not GT)")
        print(f"    branch sizes: " +
              ", ".join(f"{b.split('_')[0]}={len(branch_concepts[b])}"
                        for b in branches))
    else:
        if args.use_gt_l1:
            print("  using GT L1 grouping (--use-gt-l1)")
        else:
            print(f"  {L1_ASSIGN} not found; falling back to GT L1 grouping. "
                  "Run --stage route first for a real (non-GT) routing.")

    proposal = {"level_built": 2,
                "note": "L1 = NCCIH (given). This proposes the L2 skeleton "
                        "under each L1 branch.",
                "l1_source": ("llm_route" if branch_concepts else "gt"),
                "branches": {}}
    # route-derived mind-body concepts (to exclude from Physical/Psychological)
    mb_route = set()
    for bk, bv in branch_concepts.items():
        if bk.replace("–", "-").replace("—", "-") == "Mind-body_Therapy":
            mb_route = bv

    for b in tqdm(branches, desc="propose branches", unit="branch"):
        # concept scope: routed members if available, else GT subtree
        scope_override = branch_concepts.get(b) if branch_concepts else None

        # ---- Method B: LLM-direct (no clustering) ----
        if args.propose_mode == "llm-direct":
            if scope_override is not None:
                member_ids = sorted(scope_override - (mb_route if b in split_branches else set()))
            elif b in onto.label2id:
                member_ids = sorted(onto.subtree(b) - (mb_excl if b in split_branches else set()))
            else:
                print(f"  skip (not found): {b}"); continue
            if not member_ids:
                print(f"  skip (no concepts): {b}"); continue
            r = CM.propose_parents_llm(onto, b, member_ids, client,
                                       runs=max(1, args.runs),
                                       vote_frac=args.vote_frac,
                                       batch=args.batch,
                                       batch_threshold=args.batch_threshold)
            proposal["branches"][b] = r
            print(f"  {b:30s} -> {r['n_parents']} L2 parents "
                  f"(from {r['n_concepts']} concepts, LLM-direct x{r.get('runs',1)})")
            continue

        # ---- Method A: cluster-based (default) ----
        # if route provided concepts (scope_override), cluster those directly;
        # only require the branch to exist in the ontology when we have no scope.
        if scope_override is None and b not in onto.label2id:
            print(f"  skip (not found): {b}"); continue
        if scope_override is not None and not scope_override:
            print(f"  skip (no concepts): {b}"); continue
        excl = (mb_route if branch_concepts else mb_excl) if b in split_branches else None
        r = CM.cluster_branch(onto, b, args.embed, args.st_model, args.k,
                              args.propose, client, auto_k=(args.k == 0),
                              exclude=excl, algo=args.cluster,
                              concept_ids=scope_override)
        if r:
            proposal["branches"][b] = r
            q = r.get("quality_GTfree", {})
            print(f"  {b:30s} -> {r['png']}  (k={r['k']}, "
                  f"silhouette={q.get('silhouette')}, "
                  f"multi-parent clusters={r.get('n_multiparent_clusters')})")

    json.dump(proposal, open(PROPOSED, "w"), ensure_ascii=False, indent=2)
    print(f"\nwrote {PROPOSED} + PNGs.")
    print("PAUSED for expert review. Next:")
    print(f"  1. Expert reviews {PROPOSED} and the PNGs.")
    print(f"  2. Produce {APPROVED} (confirmed skeleton; parents = concept "
          "labels). Bootstrap a draft with:  --stage emit-approved")
    print(f"  3. Run:  python construction_agent.py --stage build ...")


def stage_emit_approved(args, onto):
    """Bootstrap a draft approved_skeleton.json from the proposal, so the expert
    edits instead of writing from scratch. The draft simply accepts every
    proposed parent; the expert then corrects it."""
    if not os.path.exists(PROPOSED):
        raise SystemExit(f"{PROPOSED} not found; run --stage propose first.")
    prop = json.load(open(PROPOSED))
    skeleton = {"l2_parents": []}
    seen = set()
    for b, r in prop["branches"].items():
        for cid, cl in r["clusters"].items():
            for g in cl["groups"]:
                p = g["parent"]
                if p and p not in seen:
                    skeleton["l2_parents"].append({"parent": p, "l1": b})
                    seen.add(p)
    json.dump(skeleton, open(APPROVED, "w"), ensure_ascii=False, indent=2)
    print(f"wrote draft {APPROVED} with {len(skeleton['l2_parents'])} parents "
          "(accepts all proposals). EDIT THIS before --stage build.")


# ===================================================================== #
#  STAGE: build  (S3-S7)                                                 #
# ===================================================================== #


def build_args():
    ap = argparse.ArgumentParser(description="SAIL Stage 2: L2 skeleton proposal")
    ap.add_argument("--stage", choices=["propose", "emit-approved"],
                    default="propose")
    ap.add_argument("--ontology", default="ontology.json")
    ap.add_argument("--branches", default="Physical_Intervention,"
                    "Psychological_Intervention,Mind–body_Therapy")
    ap.add_argument("--embed", choices=["tfidf", "st"], default="tfidf")
    ap.add_argument("--st-model", default="pritamdeka/S-BioBert-snli-multinli-stsb")
    ap.add_argument("--cluster", choices=["agg", "ap"], default="agg")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--propose", choices=["lexical", "llm"], default="lexical")
    ap.add_argument("--propose-mode", choices=["cluster", "llm-direct"],
                    default="cluster",
                    help="cluster=embedding clustering+propose; llm-direct=LLM proposes L2 parent set (method B, no clustering)")
    ap.add_argument("--runs", type=int, default=5,
                    help="llm-direct self-consistency runs (shuffle+rebatch each)")
    ap.add_argument("--vote-frac", type=float, default=0.5,
                    help="llm-direct: keep parent if proposed in >= this fraction "
                         "of runs. Lower = higher recall (fewer filtered), "
                         "higher = higher precision. 0.5=majority, 0.4=>=2/5")
    ap.add_argument("--batch", type=int, default=60,
                    help="llm-direct: concepts per batch for large branches")
    ap.add_argument("--batch-threshold", type=int, default=120,
                    help="llm-direct: branches larger than this are batched")
    ap.add_argument("--use-gt-l1", action="store_true")
    ap.add_argument("--conf-threshold", type=float, default=0.6)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--mayo", action="store_true")
    ap.add_argument("--mayo-env", default=None)
    return ap


if __name__ == "__main__":
    args = build_args().parse_args()
    onto = load_ontology(args.ontology)
    if args.stage == "emit-approved":
        stage_emit_approved(args, onto)
    else:
        stage_propose(args, onto)