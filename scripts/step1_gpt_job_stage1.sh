#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate combini_llm_construction


export ONTO="../CoMO_20260728.owl"
export ENV_FILE="/home/m319786/COMBINI/combini_llm_construction/construction/apigeex/.env"



python construction_agent.py --stage route \
  --ontology "$ONTO" \
  --mayo --model gpt-5.5 --mayo-env "$ENV_FILE" \
  --runs 1 --seeds outputs/seed_concepts_150.json \
  --conf-threshold 0.9 \
  --route-out outputs/stage1/l1_gpt.json \
  --fail-fast 5

# python construction_agent.py --stage route \
#   --ontology "$ONTO" \
#   --mayo --model gpt-5.5 --mayo-env "$ENV_FILE" \
#   --runs 5 --seeds seed_concepts_150.json \
#   --conf-threshold 0.9 \
#   --route-out l1_gpt_sc5.json \
#   --fail-fast 5