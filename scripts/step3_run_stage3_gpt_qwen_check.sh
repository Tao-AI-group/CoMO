#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate combini_llm_construction


export ONTO="../CoMO_20260728.owl"
export ENV_FILE="/home/m319786/COMBINI/combini_llm_construction/construction/apigeex/.env"


# L1
python cross_model_eval.py --ontology "$ONTO" --stage route \
  --a outputs/stage1/l1_gpt.json --b outputs/stage1/l1_qwen.json \
  --name-a gpt-5.5 --name-b qwen3 \
  --seeds outputs/seed_concepts_150.json \
  --out outputs/stage1/cross_route.json --review-csv outputs/stage1/l1_review_sheet.csv

# L2
python cross_model_eval.py --ontology "$ONTO" --stage build \
  --a outputs/stage3/build_gpt.json --b outputs/stage3/build_qwen.json \
  --skeleton outputs/stage2/approved_skeleton.json \
  --name-a gpt-5.5 --name-b qwen3 \
  --seeds outputs/seed_concepts_150.json \
  --out outputs/stage3/cross_build.json --review-csv outputs/stage3/l2_review_sheet.csv