"""
construction_agent.py — SAIL unified entry point (dispatcher).
Delegates to the standalone stage modules; each stage can also be run directly.

  --stage route         -> stage1_route.stage_route
  --stage propose       -> stage2_propose.stage_propose
  --stage emit-approved -> stage2_propose.stage_emit_approved
  --stage build         -> stage3_build.stage_build

The stage files (stage1_route.py / stage2_propose.py / stage3_build.py) are
independent and import shared code from sail_common.py.
"""
from __future__ import annotations
import argparse
from ontology_io import load_ontology
from stage1_route import stage_route
from stage2_propose import stage_propose, stage_emit_approved
from stage3_build import stage_build


def main():
    ap = argparse.ArgumentParser(description="SAIL construction agent (dispatcher)")
    ap.add_argument("--stage", choices=["route", "propose", "emit-approved",
                                        "build"], required=True)
    ap.add_argument("--ontology", default="ontology.json")
    ap.add_argument("--branches", default="Physical_Intervention,"
                    "Psychological_Intervention,Mind–body_Therapy")
    # skeleton / propose
    ap.add_argument("--embed", choices=["tfidf", "st"], default="tfidf")
    ap.add_argument("--st-model", default="pritamdeka/S-BioBert-snli-multinli-stsb")
    ap.add_argument("--cluster", choices=["agg", "ap"], default="agg")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--propose", choices=["lexical", "llm"], default="lexical")
    ap.add_argument("--propose-mode", choices=["cluster", "llm-direct"],
                    default="cluster",
                    help="cluster=embedding clustering+propose; llm-direct=LLM proposes L2 parent set (method B, no clustering)")
    ap.add_argument("--vote-frac", type=float, default=0.5,
                    help="llm-direct: keep parent if proposed in >= this fraction of runs (lower=higher recall)")
    ap.add_argument("--batch", type=int, default=60,
                    help="llm-direct: concepts per batch for large branches")
    ap.add_argument("--batch-threshold", type=int, default=120,
                    help="llm-direct: branches larger than this are batched")
    ap.add_argument("--use-gt-l1", action="store_true")
    # route / build: self-consistency + seeds
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--conf-threshold", type=float, default=0.6)
    # backend
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--route-out", default=None,
                    help="output file for stage route "
                         "(default l1_assignment.json)")
    ap.add_argument("--build-out", default=None,
                    help="output file for stage build "
                         "(default agent_build_result.json)")
    ap.add_argument("--mayo", action="store_true")
    ap.add_argument("--dual", action="store_true",
                    help="cross-model agreement: GPT (Mayo) + Qwen3 (vLLM). "
                         "Use with --runs 2 (one call per model).")
    ap.add_argument("--qwen-model", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    ap.add_argument("--qwen-base", default="http://localhost:8000/v1")
    ap.add_argument("--qwen-key", default="EMPTY")
    ap.add_argument("--mayo-env", default=None)
    ap.add_argument("--max-calls", type=int, default=0)
    ap.add_argument("--fail-fast", type=int, default=5)
    ap.add_argument("--quiet-calls", action="store_true")
    ap.add_argument("--voters", type=int, default=1)
    ap.add_argument("--tau-vote", type=float, default=0.6)
    ap.add_argument("--use-defs", action="store_true")
    args = ap.parse_args()

    onto = load_ontology(args.ontology)
    if args.stage == "route":
        stage_route(args, onto)
    elif args.stage == "propose":
        stage_propose(args, onto)
    elif args.stage == "emit-approved":
        stage_emit_approved(args, onto)
    else:
        stage_build(args, onto)


if __name__ == "__main__":
    main()