"""
extract_skeleton.py
===================
Extract the REAL L2 skeleton from the existing ontology (the expert-built
COMBINI owl). Each branch's real L2 categories = the expert's approved skeleton.

Two uses:
  1. --emit : write approved_skeleton.json from the owl's real L2 (so build can
              use the expert skeleton directly, no manual editing).
  2. --acceptance : compare an LLM-proposed skeleton against the real L2 to
              compute acceptance / rejection / omission rates.

Usage:
  # extract real L2 skeleton from owl
  python extract_skeleton.py --ontology COMBO.owl \
      --branches "Physical_Intervention,Psychological_Intervention,Mind-body_Therapy" \
      --emit --out approved_skeleton.json

  # acceptance rate of an LLM proposal vs the real L2
  python extract_skeleton.py --ontology COMBO.owl \
      --branches "..." --acceptance llmdirect_skeleton.json
"""
from __future__ import annotations
import argparse, json
from ontology_io import load_ontology
import run_pipeline as RP

ROOT = "Complementary_Medicine_Intervention"


def _norm(s):
    return s.replace("–", "-").replace("—", "-")


def real_l2_of_branch(onto, branch_label):
    """Real L2 categories under a branch (dash-insensitive).
    For an L1 branch: L2 = path[2] where path[1]==branch.
    For a mid node (e.g. mind-body sitting under Physical/Psychological):
    L2 = its direct children."""
    root = onto.lid(ROOT)
    bid = None
    for lab, i in onto.label2id.items():
        if _norm(lab) == _norm(branch_label):
            bid = i; break
    if bid is None:
        return set()
    l2 = set()
    for c in onto.subtree(onto.lab(bid)):
        for p in onto.gold_paths(c, root):
            if len(p) >= 3 and p[1] == bid:
                l2.add(onto.lab(p[2]))
    if l2:
        return l2
    return {onto.lab(k) for k in onto.children.get(bid, [])}


def proposed_parents(branch_data):
    """Extract proposed parent set from either skeleton format."""
    if "l2_parents" in branch_data:
        return set(branch_data["l2_parents"])
    parents = set()
    for cl in branch_data.get("clusters", {}).values():
        for g in cl.get("groups", []):
            parents.add(g["parent"])
    return parents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", default="ontology.json")
    ap.add_argument("--branches", default="Physical_Intervention,"
                    "Psychological_Intervention,Mind-body_Therapy")
    ap.add_argument("--emit", action="store_true",
                    help="write approved_skeleton.json from real L2")
    ap.add_argument("--acceptance", default=None,
                    help="a proposed skeleton json to score vs real L2")
    ap.add_argument("--out", default="approved_skeleton.json")
    ap.add_argument("--acc-out", default="acceptance_result.json",
                    help="where to save the acceptance-rate result")
    args = ap.parse_args()

    onto = load_ontology(args.ontology)
    branches = [b.strip() for b in args.branches.split(",")]

    # ---- extract real L2 per branch ----
    real = {b: real_l2_of_branch(onto, b) for b in branches}
    print("=== REAL L2 skeleton (from ontology) ===")
    for b in branches:
        print(f"  {b:30s}: {len(real[b])} L2 categories")

    # ---- emit approved_skeleton.json ----
    if args.emit:
        l2_parents = []
        for b in branches:
            for lab in sorted(real[b]):
                l2_parents.append({"parent": lab, "l1": b})
        json.dump({"level_built": 2, "l2_parents": l2_parents},
                  open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"\nwrote {args.out} with {len(l2_parents)} L2 parents "
              "(real expert skeleton from owl)")

    # ---- acceptance rate vs a proposed skeleton ----
    if args.acceptance:
        prop = json.load(open(args.acceptance))
        print(f"\n=== ACCEPTANCE vs proposed ({args.acceptance}) ===")
        print(f"  {'branch':<28}{'proposed':>9}{'accepted':>9}{'rejected':>9}"
              f"{'omitted':>9}{'accept%':>9}")
        tot_prop = tot_acc = tot_rej = tot_omit = 0
        per_branch = {}
        for b in branches:
            bdata = prop.get("branches", {}).get(b, {})
            pred = proposed_parents(bdata)
            gold = real[b]
            accepted = pred & gold          # LLM proposed & expert has -> accepted
            rejected = pred - gold          # LLM proposed, expert doesn't -> rejected
            omitted = gold - pred           # expert has, LLM missed -> omitted
            acc_rate = len(accepted) / len(pred) if pred else 0
            print(f"  {b:<28}{len(pred):>9}{len(accepted):>9}{len(rejected):>9}"
                  f"{len(omitted):>9}{acc_rate:>9.3f}")
            tot_prop += len(pred); tot_acc += len(accepted)
            tot_rej += len(rejected); tot_omit += len(omitted)
            per_branch[b] = {
                "n_proposed": len(pred), "n_accepted": len(accepted),
                "n_rejected": len(rejected), "n_omitted": len(omitted),
                "acceptance_rate": round(acc_rate, 3),
                "accepted": sorted(accepted), "rejected": sorted(rejected),
                "omitted": sorted(omitted)}
        print(f"  {'TOTAL':<28}{tot_prop:>9}{tot_acc:>9}{tot_rej:>9}{tot_omit:>9}"
              f"{(tot_acc/tot_prop if tot_prop else 0):>9.3f}")
        overall_acc = tot_acc / tot_prop if tot_prop else 0
        overall_omit = tot_omit / (tot_acc + tot_omit) if (tot_acc + tot_omit) else 0
        print(f"\n  Acceptance rate = {tot_acc}/{tot_prop} = "
              f"{overall_acc:.1%}  (proposed parents the expert keeps)")
        print(f"  Omission rate   = {tot_omit}/{tot_acc+tot_omit} = "
              f"{overall_omit:.1%}  (expert L2 the LLM missed)")
        print("\n  NOTE: this treats the owl's L2 as the expert skeleton. Real "
              "expert review may accept some 'rejected' ones as reasonable, so "
              "this acceptance rate is a LOWER BOUND.")
        # ---- save ----
        out = {"proposed_from": args.acceptance,
               "overall": {"n_proposed": tot_prop, "n_accepted": tot_acc,
                           "n_rejected": tot_rej, "n_omitted": tot_omit,
                           "acceptance_rate": round(overall_acc, 3),
                           "omission_rate": round(overall_omit, 3)},
               "per_branch": per_branch,
               "note": "owl L2 as expert skeleton; acceptance rate is a lower bound"}
        json.dump(out, open(args.acc_out, "w"), ensure_ascii=False, indent=2)
        print(f"\n  wrote {args.acc_out}")


if __name__ == "__main__":
    main()