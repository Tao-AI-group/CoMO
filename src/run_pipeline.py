"""
run_pipeline.py
===============
End-to-end COMBINI construction pipeline at the L1->L2 scope, comparing two
lines and showcasing multi-parent (mind-body / DAG).

Scope decision (per your choices):
  * Build depth = L1 -> L2 only (shallow, cheap). Deeper structure is collapsed
    to the L2 ancestor. Acupuncture-style deep sub-trees are left to experts.
  * L1 = the 4 NCCIH branches (anchored/given).
  * L2 = the 41 expert L2 categories are the placement SLOTS.
  * MULTI-PARENT: Mind-body_Therapy sits under BOTH Physical & Psychological;
    a concept that is both physical and psychological is placed under mind-body
    (and thus, transitively, under both L1 branches). 57 concepts are multi-L2.

Two lines:
  LINE A  (--mode given): place each concept DIRECTLY into an L2 slot, one pass
          + a consistency pass. This is the "given expert skeleton" placement.
  LINE B  (--mode build): COLD START (concepts begin unplaced under their L1
          only), then evaluator-driven ITERATIVE REFINEMENT converges them into
          L2 slots. No noise, no GT — a real from-L1 build. Stops on internal
          signal (moved fraction), GT used only as an evaluation side-channel.

  --mode both runs A and B and prints a comparison.

Backends: --mock (default, offline) / --vllm / --mayo (gpt-5.5).
Cost guards: --max-calls, --fail-fast, progress dots. GT is NEVER used to drive
the build; it is only read at evaluation time.
"""
from __future__ import annotations
import argparse, random
from collections import defaultdict

from ontology_io import load_ontology
from build_eval import MockClient, VLLMClient
# refine.py holds the OLD iterative-refinement path; the 3-stage pipeline
# (route / propose / build) does not need it, so import it lazily.

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x

ROOT_LABEL = "Complementary_Medicine_Intervention"


# ===================================================================== #
#  call monitor (failure visibility + cost guard)                       #
# ===================================================================== #
class Monitor:
    def __init__(self, max_consec_fail=5, verbose=True):
        self.ok = self.fail = self.consec = 0
        self.max_consec = max_consec_fail
        self.verbose = verbose

    def record(self, resp):
        failed = resp is None or (isinstance(resp, dict) and "_error" in resp)
        if failed:
            self.fail += 1; self.consec += 1
            if self.verbose:
                err = resp.get("_error", "?") if isinstance(resp, dict) else "None"
                print(f"\n  [call FAIL #{self.fail}] {str(err)[:120]}", flush=True)
            if self.consec >= self.max_consec:
                raise SystemExit(f"\nABORT: {self.consec} consecutive failed calls"
                                 " — check the API (python mayo_client.py <env>).")
        else:
            self.ok += 1; self.consec = 0
            if self.verbose:
                print(".", end="", flush=True)


class MonitoredClient:
    def __init__(self, inner, monitor):
        self.inner, self.monitor = inner, monitor

    def choose(self, system, user):
        r = self.inner.choose(system, user)
        self.monitor.record(r)
        return r


# ===================================================================== #
#  L2 structure from GT                                                 #
# ===================================================================== #
class L2World:
    """Everything about the L1->L2 scope, computed once from the ontology."""
    def __init__(self, onto):
        self.o = onto
        self.root = onto.lid(ROOT_LABEL)
        self.L1 = list(onto.children.get(self.root, []))
        l2 = set()
        for b in self.L1:
            for c in onto.children.get(b, []):
                # mind-body sits as a mid-node under Physical/Psychological; its
                # real L2 categories are ONE level deeper (children of it).
                if onto.lab(c).replace("–", "-") == "Mind-body_Therapy":
                    for gc in onto.children.get(c, []):
                        l2.add(gc)          # e.g. Biofeedback, Body_Psychotherapy
                else:
                    l2.add(c)
        self.L2 = sorted(l2, key=onto.lab)
        self.concepts = sorted(onto.subtree(ROOT_LABEL) - set(self.L1),
                               key=onto.lab)
        # gold L2 membership (multi for mind-body descendants)
        self._gold = {c: self._compute_gold_l2(c) for c in self.concepts}
        # which L1 each L2 sits under (mind-body -> two L1s)
        self.l2_to_l1 = {c: [p for p in onto.gold_parents(c) if p in self.L1]
                         for c in self.L2}

    def _compute_gold_l2(self, cid):
        out = set()
        for path in self.o.gold_paths(cid, self.root):
            # mind-body is a transitive mid-node: Physical/Psych -> Mind-body_Therapy
            # -> <real L2, e.g. Biofeedback> -> ... . For such concepts the L2 is
            # one level DEEPER (path[3]), not Mind-body_Therapy itself (path[2]).
            mb = (len(path) >= 3 and
                  self.o.lab(path[2]).replace("–", "-") == "Mind-body_Therapy")
            if mb and len(path) >= 4:
                out.add(path[3])            # real mind-body L2 (e.g. Biofeedback)
            elif mb and len(path) == 3:
                out.add(path[2])            # concept sits AT Mind-body_Therapy
            elif len(path) >= 3:
                out.add(path[2])            # normal L2
            elif len(path) == 2:
                out.add(path[1])            # itself an L2 -> use its own id
        return out & set(self.L2) or ({cid} if cid in set(self.L2) else set())

    def gold_l2(self, cid):
        return self._gold.get(cid, set())

    def lab(self, i):
        return self.o.lab(i)


# ===================================================================== #
#  judges                                                               #
# ===================================================================== #
class MockJudge:
    """GT-aware oracle (accuracy knob) for offline testing of the orchestration."""
    def __init__(self, world, acc=0.85, seed=0):
        self.w, self.acc, self.rng = world, acc, random.Random(seed)

    def place(self, concept, slots):
        gold = self.w.gold_l2(concept) & set(slots)
        if gold and self.rng.random() < self.acc:
            return set(list(gold))          # multi-parent aware
        return {self.rng.choice(slots)} if slots else set()


class LLMJudge:
    def __init__(self, world, client):
        self.w, self.c = world, client

    def place(self, concept, slots):
        sys = ('Assign the concept to the complementary-medicine category it '
               'IS-A. A concept that is BOTH physical and psychological belongs '
               'to Mind-body_Therapy. You may pick MORE THAN ONE if it genuinely '
               'belongs to several. Respond ONLY JSON: {"categories":["<label>"]}')
        lines = [f"Concept: {self.w.lab(concept)}", "Categories:"]
        lines += [f"  - {self.w.lab(s)}" for s in slots]
        r = self.c.choose(sys, "\n".join(lines)) or {}
        picked = set(r.get("categories", []) or [])
        return {s for s in slots if self.w.lab(s) in picked}


# ===================================================================== #
#  LINE A: given-skeleton direct placement                              #
# ===================================================================== #
def line_a(world, judge, monitor, max_calls):
    assign = {}
    for c in tqdm(world.concepts, desc="lineA place", unit="c", leave=False):
        if max_calls and monitor and monitor.ok + monitor.fail >= max_calls:
            break
        assign[c] = judge.place(c, world.L2)
    return assign


# ===================================================================== #
#  LINE B: cold-start iterative refinement                              #
# ===================================================================== #
def line_b(world, judge, voters, use_defs, max_rounds, stop_frac,
           monitor, max_calls, tau_vote=0.6):
    o = world.root
    # cold start: everyone unplaced -> put each concept into ONE L2 slot by a
    # first placement pass (no GT, no noise).
    assign = {}
    for c in tqdm(world.concepts, desc="lineB cold-start", unit="c", leave=False):
        if max_calls and monitor and monitor.ok + monitor.fail >= max_calls:
            break
        p = judge.place(c, world.L2)
        assign[c] = p if p else {world.L2[0]}

    history = defaultdict(list)
    print(f"\n{'round':>5}{'moved':>7}{'problems':>10}{'oscill':>8}"
          f"{'  acc(GT,side)':>16}")
    print("-" * 46)
    for rnd in range(1, max_rounds + 1):
        # group by (any) assigned slot
        groups = defaultdict(list)
        for c, slots in assign.items():
            for s in slots:
                groups[s].append(c)

        moved = oscill = problems = 0
        budget_hit = False
        todo = [(s, m) for s, m in groups.items() if len(m) >= 2]
        for slot, members in tqdm(todo, desc=f"round {rnd}", unit="cat",
                                  leave=False):
            if max_calls and monitor and monitor.ok + monitor.fail >= max_calls:
                budget_hit = True; break
            # evaluator voting -> confident outliers
            from refine import find_outliers      # lazy: old path only
            counts = defaultdict(int)
            for v in voters:
                for m in find_outliers(world.o, slot, members, [v], use_defs):
                    counts[m] += 1
            confident = {m for m, k in counts.items() if k / len(voters) >= tau_vote}
            problems += len(confident)
            for m in confident:
                newp = judge.place(m, [s for s in world.L2])
                if not newp:
                    continue
                if newp != assign[m]:
                    if any(p in history[m] for p in newp):
                        oscill += 1
                    history[m].append(tuple(sorted(assign[m])))
                    assign[m] = newp
                    moved += 1

        # side-channel accuracy (NOT used for decisions)
        correct = sum(1 for c in world.concepts
                      if assign[c] & world.gold_l2(c))
        acc = correct / len(world.concepts)
        print(f"{rnd:>5}{moved:>7}{problems:>10}{oscill:>8}{acc:>16.3f}")

        if budget_hit:
            print("  -> hit --max-calls; stopping"); break
        if moved == 0 or moved / len(world.concepts) < stop_frac:
            print(f"  -> converged (moved={moved}), internal-signal stop"); break
    return assign


# ===================================================================== #
#  evaluation (L2, multi-parent aware)                                  #
# ===================================================================== #
def evaluate(world, assign, label):
    n = len(world.concepts)
    exact = mp_ok = mp_tot = 0
    inter = pred_sz = gold_sz = 0
    for c in world.concepts:
        pred = assign.get(c, set())
        gold = world.gold_l2(c)
        if pred & gold:                      # hit any gold L2
            exact += 1
        inter += len(pred & gold); pred_sz += len(pred); gold_sz += len(gold)
        if len(gold) > 1:                    # multi-parent concept
            mp_tot += 1
            if gold <= pred:                 # recovered ALL its parents
                mp_ok += 1
    P = inter / pred_sz if pred_sz else 0
    R = inter / gold_sz if gold_sz else 0
    F = 2 * P * R / (P + R) if (P + R) else 0
    return {
        "line": label, "n": n,
        "l2_accuracy": round(exact / n, 3),
        "setP": round(P, 3), "setR": round(R, 3), "setF1": round(F, 3),
        "multiparent_full_recovery": (f"{mp_ok}/{mp_tot}"
                                      f" ({mp_ok/mp_tot:.0%})" if mp_tot else "n/a"),
    }


def print_table(rows):
    print(f"\n{'line':<8}{'L2 acc':>9}{'setP':>8}{'setR':>8}{'setF1':>8}"
          f"{'mind-body multi':>18}")
    print("-" * 59)
    for r in rows:
        print(f"{r['line']:<8}{r['l2_accuracy']:>9}{r['setP']:>8}{r['setR']:>8}"
              f"{r['setF1']:>8}{r['multiparent_full_recovery']:>18}")


# ===================================================================== #
def build_backend(args, world):
    if args.mock:
        voters = [MockClient(accuracy=0.85, seed=i) for i in range(args.voters)]
        judge = MockJudge(world, acc=0.85)
        return judge, voters, None
    monitor = Monitor(max_consec_fail=args.fail_fast, verbose=not args.quiet_calls)
    if getattr(args, "dual", False):
        # cross-model agreement: GPT (Mayo) + Qwen3 (local vLLM), round-robin
        # over `runs`, so the existing vote machinery yields cross-model
        # agreement as the confidence signal. Use --runs 2 for one call each.
        from mayo_client import MayoClient
        from qwen_client import QwenClient
        from dual_client import DualClient

        def mk():
            gpt = MayoClient(engine=args.model, env_path=args.mayo_env)
            qwen = QwenClient(model=args.qwen_model,
                              base_url=args.qwen_base,
                              api_key=args.qwen_key)
            return MonitoredClient(
                DualClient([gpt, qwen], names=[args.model, "qwen3"]), monitor)
    elif args.mayo:
        from mayo_client import MayoClient
        mk = lambda: MonitoredClient(MayoClient(engine=args.model,
                                                env_path=args.mayo_env), monitor)
    else:
        mk = lambda: MonitoredClient(VLLMClient(args.model, args.api_base,
                                                args.api_key), monitor)
    voters = [mk() for _ in range(args.voters)]
    judge = LLMJudge(world, mk())
    return judge, voters, monitor


def main():
    ap = argparse.ArgumentParser(description="COMBINI L1->L2 pipeline")
    ap.add_argument("--ontology", default="ontology.json")
    ap.add_argument("--mode", choices=["given", "build", "both"], default="both")
    ap.add_argument("--use-defs", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--stop-frac", type=float, default=0.02)
    ap.add_argument("--voters", type=int, default=1)
    ap.add_argument("--tau-vote", type=float, default=0.6)
    # backend
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
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
    args = ap.parse_args()
    if not (args.mock or args.mayo or args.model):
        args.mock = True

    onto = load_ontology(args.ontology)
    world = L2World(onto)
    print(f"L1 branches={len(world.L1)}  L2 categories={len(world.L2)}  "
          f"concepts={len(world.concepts)}  "
          f"multiparent(concepts)={sum(1 for c in world.concepts if len(world.gold_l2(c))>1)}")

    rows = []
    if args.mode in ("given", "both"):
        judge, voters, monitor = build_backend(args, world)
        print("\n=== LINE A: given-skeleton direct placement ===")
        a = line_a(world, judge, monitor, args.max_calls)
        rows.append(evaluate(world, a, "given"))
    if args.mode in ("build", "both"):
        judge, voters, monitor = build_backend(args, world)
        print("\n=== LINE B: cold-start iterative refinement ===")
        b = line_b(world, judge, voters, args.use_defs, args.max_rounds,
                   args.stop_frac, monitor, args.max_calls, args.tau_vote)
        rows.append(evaluate(world, b, "build"))

    print_table(rows)
    print("\nNote: GT used ONLY for evaluation; the build loop stops on the "
          "internal moved-fraction signal, never on GT.")


if __name__ == "__main__":
    main()