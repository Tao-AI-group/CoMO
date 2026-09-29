"""
build_eval.py
-------------
LLM-assisted ontology CONSTRUCTION pipeline + evaluation, for one branch
(default: Complementary_Medicine_Intervention).

Idea
====
The expert-reviewed ontology is the GROUND TRUTH. We start from only the flat
bag of concept labels (hierarchy stripped) and rebuild the placement the way the
paper describes it:

  Step 1 (skeleton):   the human-authored category nodes are *given* per level
                       (coarse -> fine), seeded from NCCIH top-level branches.
  Step 2 (placement):  for each concept, an LLM chooses which child-category it
                       belongs under, descending the skeleton level by level
                       (staged elaboration). This rebuilds every parent->child
                       edge.
  Step 3 (refinement): in the real project this is the expert pass; here the
                       expert answer already exists (the GT), so this script
                       just SCORES the LLM placement against it.

We run the cascade top-down so it produces "build the tree step by step" AND a
per-level accuracy, which is what an ontology-construction pipeline needs to
prove it helps.

Models are served via a vLLM OpenAI-compatible API (medgemma / llama4 / qwen3).
A --mock backend lets you test the whole harness with no endpoint.

Outputs (to --out-dir):
  predictions.jsonl   one row per concept: gold parent(s), predicted parent, path
  metrics.json        branch acc, exact-parent acc, hierarchical P/R/F, per-depth
  confusion_top.csv   top-level (4-branch) confusion matrix
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from typing import Optional

from ontology_io import Ontology, load_ontology

ATTACH_HERE = "__ATTACH_HERE__"


# ====================================================================== #
#  LLM backends                                                          #
# ====================================================================== #
def balanced_json(text: str) -> Optional[dict]:
    """Brace-balanced JSON extraction (robust to preamble / code fences)."""
    start = text.find("{")
    if start < 0:
        return None
    depth, instr, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
        else:
            if c == '"':
                instr = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


class VLLMClient:
    """OpenAI-compatible chat client (vLLM). Retry + timeout."""
    def __init__(self, model: str, api_base: str, api_key: str = "EMPTY",
                 timeout: int = 60, retries: int = 3, temperature: float = 0.0):
        from openai import OpenAI
        self.client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout)
        self.model = model
        self.retries = retries
        self.temperature = temperature

    def choose(self, system: str, user: str) -> dict:
        last = None
        for attempt in range(self.retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                )
                txt = r.choices[0].message.content
                obj = balanced_json(txt)
                if obj is not None:
                    return obj
                last = f"unparseable: {txt[:160]}"
            except Exception as e:  # noqa: BLE001
                last = str(e)
            time.sleep(1.5 * (attempt + 1))
        return {"choice": None, "_error": last}


class MockClient:
    """
    Deterministic-ish stand-in for testing the harness with no endpoint.
    Returns the GOLD child with probability `accuracy`, else a random sibling.
    Lets you validate metrics end-to-end and see plausible numbers.
    """
    def __init__(self, accuracy: float = 0.85, seed: int = 0):
        self.accuracy = accuracy
        self.rng = random.Random(seed)
        self._gold = None  # set by pipeline right before each call

    def choose(self, system: str, user: str) -> dict:
        gold, candidates = self._gold
        if self.rng.random() < self.accuracy:
            return {"choice": gold, "confidence": 0.9}
        pool = [c for c in candidates if c != gold] or [gold]
        return {"choice": self.rng.choice(pool), "confidence": 0.4}


# ====================================================================== #
#  Prompting                                                             #
# ====================================================================== #
SYSTEM = (
    "You are an expert in complementary & integrative medicine helping to build "
    "an ontology. Given a concept and a parent category with its candidate "
    "sub-categories, decide which single sub-category the concept belongs under. "
    "If the concept is itself an instance of the parent and does NOT fit any "
    "narrower sub-category, choose ATTACH_HERE. "
    "Respond with ONLY a JSON object: "
    '{"choice": "<exact sub-category label or ATTACH_HERE>", "confidence": 0-1}. '
    "No prose, no markdown."
)


def build_user_prompt(onto: Ontology, concept_id: str, parent_id: str,
                      candidate_ids: list[str], use_defs: bool) -> str:
    n = onto.nodes[concept_id]
    lines = [f"CONCEPT: {n.label}"]
    if n.alt:
        lines.append("  also known as: " + ", ".join(n.alt[:6]))
    if use_defs and n.defn:
        lines.append("  gloss: " + n.defn[:300])
    lines.append(f"\nPARENT CATEGORY: {onto.lab(parent_id)}")
    if use_defs and onto.nodes[parent_id].defn:
        lines.append("  " + onto.nodes[parent_id].defn[:200])
    lines.append("\nCANDIDATE SUB-CATEGORIES:")
    for cid in candidate_ids:
        c = onto.nodes[cid]
        d = f" — {c.defn[:120]}" if (use_defs and c.defn) else ""
        lines.append(f"  - {c.label}{d}")
    lines.append(f"  - ATTACH_HERE (concept attaches directly under "
                 f"{onto.lab(parent_id)})")
    lines.append('\nReturn JSON: {"choice": "...", "confidence": 0-1}')
    return "\n".join(lines)


# ====================================================================== #
#  Pipeline: top-down cascade placement                                 #
# ====================================================================== #
def category_children(onto: Ontology, node_id: str, scope: set[str]) -> list[str]:
    """In-scope children of node that are themselves categories (skeleton slots)."""
    return [c for c in onto.children.get(node_id, [])
            if c in scope and onto.is_category(c)]


def place_concept(onto: Ontology, concept_id: str, root: str, scope: set[str],
                  client, use_defs: bool, gold_parents: set[str]) -> dict:
    """
    Descend root -> ... choosing a child-category at each level until the model
    says ATTACH_HERE (or runs out of sub-categories). Returns predicted path +
    predicted parent. The candidate set at each node = that node's gold
    child-categories (skeleton is human-authored = given), so the LLM only
    decides PLACEMENT, matching the paper's Step 1 / Step 2 split.
    """
    path = [root]
    cur = root
    steps = []
    while True:
        cands = category_children(onto, cur, scope)
        # the concept itself is a category -> exclude it from its own candidates
        cands = [c for c in cands if c != concept_id]
        if not cands:
            break
        # gold action at this node (for mock + per-level oracle scoring)
        gold_child = None
        for c in cands:
            if c == concept_id:
                continue
            # does the gold path of concept pass through c?
            if c in onto.ancestors(concept_id, root):
                gold_child = c
                break
        gold_label = onto.lab(gold_child) if gold_child else ATTACH_HERE

        if isinstance(client, MockClient):
            client._gold = (gold_label, [onto.lab(c) for c in cands] + [ATTACH_HERE])

        user = build_user_prompt(onto, concept_id, cur, cands, use_defs)
        resp = client.choose(SYSTEM, user)
        choice = (resp or {}).get("choice")
        steps.append({"node": onto.lab(cur), "gold": gold_label,
                      "pred": choice, "conf": (resp or {}).get("confidence")})

        if choice == ATTACH_HERE or choice == "ATTACH_HERE" or choice is None:
            break
        # map predicted label back to an id among candidates
        chosen = next((c for c in cands if onto.lab(c) == choice), None)
        if chosen is None:
            break                      # hallucinated / off-list -> stop here
        path.append(chosen)
        cur = chosen
        if cur == concept_id:          # safety
            break

    pred_parent = path[-1]
    return {
        "concept": onto.lab(concept_id),
        "concept_id": concept_id,
        "pred_parent": onto.lab(pred_parent),
        "pred_parent_id": pred_parent,
        "pred_path": [onto.lab(p) for p in path],
        "gold_parents": sorted(onto.lab(p) for p in gold_parents),
        "steps": steps,
    }


# ====================================================================== #
#  Metrics                                                               #
# ====================================================================== #
def top_branch(onto: Ontology, nid: str, root: str) -> Optional[str]:
    """The level-1 branch (direct child of root) an ancestor chain goes through."""
    for path in onto.gold_paths(nid, root):
        if len(path) >= 2:
            return onto.lab(path[1])
    return None


def pred_top_branch(pred_path: list[str]) -> Optional[str]:
    return pred_path[1] if len(pred_path) >= 2 else None


def evaluate(onto: Ontology, root: str, preds: list[dict],
             concept_ids: list[str]) -> dict:
    n = len(preds)
    branch_hit = exact_parent = path_exact = 0
    hP_num = hP_den = hR_num = hR_den = 0
    by_depth = defaultdict(lambda: [0, 0])  # depth -> [hit, total] (exact parent)
    conf = defaultdict(lambda: defaultdict(int))  # gold_branch -> pred_branch

    for p, cid in zip(preds, concept_ids):
        gold_parents = set(onto.gold_parents(cid))
        pred_parent_id = p["pred_parent_id"]

        # exact parent (multi-parent => credit if matches ANY gold parent)
        ok_parent = pred_parent_id in gold_parents
        exact_parent += ok_parent

        # depth bucket (length of shortest gold path)
        depth = min(len(pp) for pp in onto.gold_paths(cid, root))
        by_depth[depth][1] += 1
        by_depth[depth][0] += ok_parent

        # top-level branch
        gb = top_branch(onto, cid, root)
        pb = pred_top_branch(p["pred_path"])
        branch_hit += (gb is not None and gb == pb)
        if gb:
            conf[gb][pb or "<none>"] += 1

        # full path exact (against any gold line)
        gold_label_paths = [[onto.lab(x) for x in pp]
                            for pp in onto.gold_paths(cid, root)]
        path_exact += (p["pred_path"] in gold_label_paths)

        # hierarchical P/R/F  (Silla & Freitas): ancestor-set overlap
        gold_anc = onto.ancestors(cid, root)
        pred_anc = set(p["pred_path"][1:-1])  # category ancestors, label-space
        pred_anc = {onto.lid(l) for l in pred_anc if l in onto.label2id}
        inter = len(pred_anc & gold_anc)
        hP_num += inter; hP_den += len(pred_anc)
        hR_num += inter; hR_den += len(gold_anc)

    hP = hP_num / hP_den if hP_den else 0.0
    hR = hR_num / hR_den if hR_den else 0.0
    hF = 2 * hP * hR / (hP + hR) if (hP + hR) else 0.0

    return {
        "n_concepts": n,
        "top_branch_accuracy": round(branch_hit / n, 4),
        "exact_parent_accuracy": round(exact_parent / n, 4),
        "full_path_exact_match": round(path_exact / n, 4),
        "hierarchical_precision": round(hP, 4),
        "hierarchical_recall": round(hR, 4),
        "hierarchical_f1": round(hF, 4),
        "exact_parent_by_depth": {
            d: {"acc": round(h / t, 4) if t else None, "n": t}
            for d, (h, t) in sorted(by_depth.items())
        },
        "_confusion_top": {g: dict(row) for g, row in conf.items()},
    }


def write_confusion_csv(conf: dict, branches: list[str], path: str):
    cols = branches + ["<none>"]
    with open(path, "w") as f:
        f.write("gold\\pred," + ",".join(cols) + "\n")
        for g in branches:
            row = conf.get(g, {})
            f.write(g + "," + ",".join(str(row.get(c, 0)) for c in cols) + "\n")


# ====================================================================== #
#  Orchestration                                                         #
# ====================================================================== #
def main():
    ap = argparse.ArgumentParser(description="LLM-assisted ontology build + eval")
    ap.add_argument("--ontology", default="ontology.json")
    ap.add_argument("--branch", default="Complementary_Medicine_Intervention")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--use-defs", action="store_true",
                    help="feed gold glosses to the LLM (note: mild leakage; "
                         "default off = realistic from-scratch setting)")
    ap.add_argument("--limit", type=int, default=0, help="cap #concepts (debug)")
    # backend
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mock-acc", type=float, default=0.85)
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    onto = load_ontology(args.ontology)
    root = onto.lid(args.branch)
    scope = onto.subtree(args.branch) | {root}

    # concepts to place = every in-scope node except the root
    concept_ids = sorted(scope - {root}, key=lambda i: onto.lab(i))
    if args.limit:
        concept_ids = concept_ids[:args.limit]

    branches = [onto.lab(c) for c in onto.children.get(root, []) if c in scope]

    if args.mock:
        client = MockClient(accuracy=args.mock_acc)
        backend = f"MOCK(acc={args.mock_acc})"
    else:
        client = VLLMClient(args.model, args.api_base, args.api_key,
                            temperature=args.temperature)
        backend = f"vLLM:{args.model}"

    print(f"branch={args.branch}  concepts={len(concept_ids)}  "
          f"top-branches={len(branches)}  backend={backend}")

    preds = []
    t0 = time.time()
    for i, cid in enumerate(concept_ids, 1):
        gold_parents = set(onto.gold_parents(cid))
        preds.append(place_concept(onto, cid, root, scope, client,
                                   args.use_defs, gold_parents))
        if i % 50 == 0:
            print(f"  placed {i}/{len(concept_ids)}  "
                  f"({(time.time()-t0)/i:.2f}s/concept)")

    metrics = evaluate(onto, root, preds, concept_ids)
    conf = metrics.pop("_confusion_top")

    with open(os.path.join(args.out_dir, "predictions.jsonl"), "w") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"backend": backend, "branch": args.branch,
                   "use_defs": args.use_defs, **metrics}, f, indent=2)
    write_confusion_csv(conf, branches,
                        os.path.join(args.out_dir, "confusion_top.csv"))

    print("\n=== METRICS ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nwrote -> {args.out_dir}/{{predictions.jsonl,metrics.json,"
          f"confusion_top.csv}}")


if __name__ == "__main__":
    main()