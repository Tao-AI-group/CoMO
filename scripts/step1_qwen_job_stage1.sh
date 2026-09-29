#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate combini_llm_construction

export ONTO="../CoMO_20260728.owl"
export QWEN_BASE="http://127.0.0.1:8000/v1"


curl -s "$QWEN_BASE/models" >/dev/null || { echo "vLLM can not be connected"; exit 1; }


python construction_agent.py --stage route \
  --ontology "$ONTO" \
  --model "Qwen/Qwen3-235B-A22B-Instruct-2507" \
  --api-base "$QWEN_BASE" --api-key EMPTY \
  --runs 1 --seeds outputs/seed_concepts_150.json \
  --conf-threshold 0.9 \
  --route-out outputs/stage1/l1_qwen.json \
  --fail-fast 5