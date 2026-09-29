#!/bin/bash
# =====================================================================
# Build the 150-concept evaluation seed set
# =====================================================================
#   - all concepts under Dietary_Intervention and Natural_Product_Intervention
#   - random concepts from Physical / Psychological / Mind-body to reach 150
#
# Usage:
#   bash scripts/make_seeds.sh            # writes data/seed_concepts_150.json
#   bash scripts/make_seeds.sh --force    # overwrite an existing seed file
#
# The paper run used: ontology COMBO_20260605.owl, total 150, random seed 0,
# no excluded concepts. With the same ontology these defaults reproduce the
# archived seed set exactly. If the ontology version changes, the sample
# changes too, so keep the archived file when reproducing the paper.
# ---------------------------------------------------------------------

set -euo pipefail
cd "$(dirname "$0")/.."            # always run from the repository root

# ------------------------- CONFIG (override via env vars) ------------
ONTO="${ONTO:-/home/m319786/COMBINI/combini_llm_construction/for_github/CoMO/CoMO_20260728.owl}"
TOTAL="${TOTAL:-150}"
RANDOM_SEED="${RANDOM_SEED:-0}"
EXCLUDE="${EXCLUDE:-}"             # optional JSON: {"concepts": [...]}
OUT="${OUT:-outputs/seed_concepts_${TOTAL}.json}"

# ------------------------- checks ------------------------------------
if [ ! -f "$ONTO" ]; then
  echo "ERROR: ontology file not found: $ONTO"
  echo "       download it from https://github.com/Tao-AI-group/COMBINI"
  exit 1
fi

if [ -f "$OUT" ] && [ "${1:-}" != "--force" ]; then
  echo "ERROR: $OUT already exists (archived seed set)."
  echo "       Use --force to overwrite, or set OUT to another path, e.g.:"
  echo "       OUT=data/seed_concepts_new.json bash scripts/make_seeds.sh"
  exit 1
fi

# ------------------------- run ---------------------------------------
ARGS=(--ontology "$ONTO" --total "$TOTAL" --seed "$RANDOM_SEED" --out "$OUT")
[ -n "$EXCLUDE" ] && ARGS+=(--exclude "$EXCLUDE")

echo "### Building seed set: total=$TOTAL, random seed=$RANDOM_SEED"
python src/make_seeds_150.py "${ARGS[@]}"
echo "  -> $OUT"
echo "  Copy this file to every machine that runs a model."