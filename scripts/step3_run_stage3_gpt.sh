#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate combini_llm_construction


export ONTO="../CoMO_20260728.owl"
export ENV_FILE="/home/m319786/COMBINI/combini_llm_construction/construction/apigeex/.env"

python construction_agent.py --stage build \
  --ontology "$ONTO" \
  --mayo --model gpt-5.5 --mayo-env "$ENV_FILE" \
  --runs 1 --seeds outputs/seed_concepts_150.json \
  --conf-threshold 0.9 \
  --build-out outputs/stage3/build_gpt.json --fail-fast 5