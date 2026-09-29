"""
make_seeds_150.py
=================
Build the evaluation seed set:
  - ALL concepts under Dietary_Intervention
  - ALL concepts under Natural_Product_Intervention
  - random concepts from the remaining branches (Physical / Psychological /
    Mind-body) to reach --total (default 150)

Rationale: Dietary and Natural Product are small branches; sampling them would
leave 1-3 instances per L2 category and make per-label F1 meaningless. Taking
them whole fixes that, and the random remainder keeps the set unbiased for the
two large branches.

Usage:
  python make_seeds_150.py --ontology COMBO.owl --total 150 \
      --exclude fewshot_concepts.json --out seed_concepts_150.json
"""
from __future__ import annotations
import argparse, json, random
from ontology_io import load_ontology
import run_pipeline as RP


SMALL_BRANCHES = ["Dietary_Intervention", "Natural_Product_Intervention"]


def norm(s: str) -> str:
    return s.replace("\u2013", "-").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", default="ontology.json")
    ap.add_argument("--total", type=int, default=150)
    ap.add_argument("--small-branches", default=",".join(SMALL_BRANCHES),
                    help="branches taken in FULL")
    ap.add_argument("--exclude", default=None,
                    help="JSON file with {'concepts': [...]} to exclude "
                         "(e.g. few-shot examples used in prompts)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="seed_concepts_150.json")
    args = ap.parse_args()

    onto = load_ontology(args.ontology)
    world = RP.L2World(onto)
    rng = random.Random(args.seed)

    lab = {c: world.lab(c) for c in world.concepts}
    small = [norm(b) for b in args.small_branches.split(",")]

    # --- excluded (few-shot) concepts -------------------------------- #
    excluded = set()
    if args.exclude:
        try:
            ex = json.load(open(args.exclude))
            excluded = set(ex.get("concepts", ex.get("seed_concepts", [])))
            print(f"  excluding {len(excluded)} concepts from {args.exclude}")
        except Exception as e:                                   # noqa: BLE001
            print(f"  WARNING: could not read --exclude ({e}); continuing")

    # --- which branch(es) does each concept belong to? ---------------- #
    root = onto.lid(RP.ROOT_LABEL)

    def branches_of(cid):
        out = set()
        for path in onto.gold_paths(cid, root):
            if len(path) >= 2:
                out.add(norm(onto.lab(path[1])))
        return out

    full_pool, rest_pool = [], []
    for c in world.concepts:
        name = lab[c]
        if name in excluded:
            continue
        brs = branches_of(c)
        if brs & set(small):
            full_pool.append(name)
        else:
            rest_pool.append(name)

    full_pool = sorted(set(full_pool))
    rest_pool = sorted(set(rest_pool))

    n_need = max(0, args.total - len(full_pool))
    if n_need > len(rest_pool):
        print(f"  WARNING: only {len(rest_pool)} other concepts available; "
              f"seed set will be {len(full_pool) + len(rest_pool)}")
        n_need = len(rest_pool)
    sampled = sorted(rng.sample(rest_pool, n_need))

    seeds = sorted(set(full_pool) | set(sampled))

    # --- report composition ------------------------------------------ #
    from collections import Counter
    comp = Counter()
    for name in seeds:
        cid = onto.lid(name)
        for b in branches_of(cid):
            comp[b] += 1

    print(f"\n=== seed set: {len(seeds)} concepts ===")
    print(f"  full-coverage branches ({', '.join(small)}): {len(full_pool)}")
    print(f"  sampled from the rest:                        {len(sampled)}")
    print("  branch composition (multi-label, sums > total):")
    for b, n in comp.most_common():
        print(f"    {b:32s} {n}")

    json.dump({"seed_concepts": seeds,
               "n": len(seeds),
               "full_branches": small,
               "n_full": len(full_pool),
               "n_sampled": len(sampled),
               "excluded": sorted(excluded),
               "random_seed": args.seed},
              open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()