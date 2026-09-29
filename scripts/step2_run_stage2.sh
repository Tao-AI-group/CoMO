#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate combini_llm_construction


export ONTO="../CoMO_20260728.owl"
export ENV_FILE="/home/m319786/COMBINI/combini_llm_construction/construction/apigeex/.env"

# cp l1_merged.json l1_assignment.json

python construction_agent.py --stage propose \
  --ontology "$ONTO" \
  --mayo --model gpt-5.5 --mayo-env "$ENV_FILE" \
  --propose llm \
  --propose-mode llm-direct \
  --runs 1 \
  --branches "Physical_Intervention,Psychological_Intervention,Mind-body_Therapy"



python extract_skeleton.py --ontology "$ONTO" \
  --branches "Physical_Intervention,Psychological_Intervention,Mind-body_Therapy" \
  --acceptance outputs/stage2/proposed_skeleton.json \
  --acc-out outputs/stage2/acceptance_result.json


python extract_skeleton.py --ontology "$ONTO" \
  --branches "Physical_Intervention,Psychological_Intervention,Mind-body_Therapy" \
  --emit --out outputs/stage2/approved_skeleton.json