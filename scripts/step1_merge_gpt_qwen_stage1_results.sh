source ~/anaconda3/etc/profile.d/conda.sh
conda activate combini_llm_construction

export ONTO="../CoMO_20260728.owl"

python cross_model_eval.py --ontology "$ONTO" --stage route \
  --a outputs/stage1/l1_gpt.json --b outputs/stage1/l1_qwen.json \
  --name-a gpt-5.5 --name-b qwen3 \
  --seeds outputs/seed_concepts_150.json --out outputs/stage1/cross_route.json


python merge_l1.py --ontology "$ONTO" \
  --a outputs/stage1/l1_gpt.json --b outputs/stage1/l1_qwen.json \
  --resolve gold --out outputs/stage1/l1_merged.json

cp outputs/stage1/l1_merged.json outputs/stage2/l1_assignment.json   