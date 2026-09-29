#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate combini_llm_construction

export ONTO="../CoMO_20260728.owl"
export ENV_FILE="/home/m319786/COMBINI/combini_llm_construction/construction/apigeex/.env"

python construction_agent.py --stage build \
  --ontology "$ONTO" \
  --model "Qwen/Qwen3-235B-A22B-Instruct-2507" \
  --api-base "http://127.0.0.1:8000/v1" --api-key EMPTY \
  --runs 1 --seeds outputs/seed_concepts_150.json \
  --conf-threshold 0.9 \
  --build-out outputs/stage3/build_qwen.json --fail-fast 5