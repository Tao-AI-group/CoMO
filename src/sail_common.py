"""
sail_common.py
==============
Shared utilities for the SAIL construction agent stages
(stage1_route / stage2_propose / stage3_build).

Contains: constants, tqdm shim, multi-label evaluation, gold-label helpers.
Each stage file imports from here so logic is not duplicated.
"""
from __future__ import annotations
import json, os

ROOT_LABEL = "Complementary_Medicine_Intervention"
PROPOSED = "outputs/stage2/proposed_skeleton.json"
APPROVED = "outputs/stage2/approved_skeleton.json"
L1_ASSIGN = "outputs/stage2/l1_assignmen.json"
BUILD_RESULT = "outputs/stage3/agent_build_result.json"

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x


def evaluate_multilabel(Y_true, Y_pred, label_names):
    """Standard multi-label metrics via sklearn.
    Y_true, Y_pred: (n_samples, n_labels) 0/1 matrices."""
    from sklearn.metrics import (accuracy_score, hamming_loss, f1_score,
                                 precision_score, recall_score)
    m = {
        # 1. Subset Accuracy (Exact Match) — whole label set must match
        "subset_accuracy": round(float(accuracy_score(Y_true, Y_pred)), 3),
        # 2. Hamming Loss — per-label error rate (lower better)
        "hamming_loss": round(float(hamming_loss(Y_true, Y_pred)), 3),
        # 3. Micro / Macro / Weighted F1 (label-based)
        "micro_f1": round(float(f1_score(Y_true, Y_pred, average="micro",
                                         zero_division=0)), 3),
        "macro_f1": round(float(f1_score(Y_true, Y_pred, average="macro",
                                         zero_division=0)), 3),
        "weighted_f1": round(float(f1_score(Y_true, Y_pred, average="weighted",
                                            zero_division=0)), 3),
        # 4. Example-based (sample-averaged) P/R/F1
        "example_P": round(float(precision_score(Y_true, Y_pred, average="samples",
                                                 zero_division=0)), 3),
        "example_R": round(float(recall_score(Y_true, Y_pred, average="samples",
                                              zero_division=0)), 3),
        "example_F1": round(float(f1_score(Y_true, Y_pred, average="samples",
                                           zero_division=0)), 3),
    }
    # 5. Per-label P/R/F1
    per = {}
    P = precision_score(Y_true, Y_pred, average=None, zero_division=0)
    R = recall_score(Y_true, Y_pred, average=None, zero_division=0)
    Fs = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    for i, lab in enumerate(label_names):
        per[lab] = {"P": round(float(P[i]), 3), "R": round(float(R[i]), 3),
                    "F1": round(float(Fs[i]), 3)}
    m["per_label"] = per
    return m


def gold_l1(world, cid):
    """GT L1 branch(es) of a concept — the 4 NCCIH branches only (multi-label).
    mind-body is NOT an L1 label; it is derived after routing."""
    out = set()
    for path in world.o.gold_paths(cid, world.root):
        if len(path) >= 2:
            out.add(path[1])             # depth-1 node = L1 branch
    return out & set(world.L1)