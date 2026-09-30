# CoMO: A Complementary Medicine Ontology Developed with Large Language Model Assistance 

# Overview

The Complementary Medicine Ontology (CoMO) is a FAIR-compliant ontology for the standardized representation of complementary medicine concepts and their semantic relationships. It is designed to support semantic interoperability, knowledge integration, and computational research in complementary and integrative healthcare.

CoMO contains 597 classes, 5 object properties, 3 data properties, and 13 annotation properties across three major semantic domains: complementary medicine interventions, products, and systems. It provides hierarchical and semantic representations of complementary medicine concepts, along with annotations such as preferred labels, synonyms, definitions, comments, and mappings to UMLS concepts and semantic types.

CoMO was developed using a level-wise, LLM-assisted, human-in-the-loop framework that combines large language models with expert review and adjudication. Its semantic validity, coverage, and coherence were evaluated through expert assessment and corpus-based evaluation using PubMed literature.

The overview of CoMO development and evaluation framework is illustrated below.

<p align="center">
  <img src="Figures/Overview_of_CoMO_development_and_evaluation_framework.png" alt="Overview of CoMO Development and Evaluation Framework" width="600">
</p>

# Ontology Development
CoMO was developed iteratively across hierarchical levels using a large language model (LLM)-assisted, human-in-the-loop framework. At each level, an LLM (GPT 5.5) identified candidate hypernym categories, which were reviewed and validated by subject matter experts. Two LLMs (GPT 5.5 and Qwen3-225B-A22B-Instruct) then independently classified concepts into the validated categories, with disagreements routed for expert adjudication.

<p align="center">
  <img src="Figures/LLM-assisted_ontology_construction_framework.png" alt="LLM assisted ontology construction Framework" width="600">
</p>


# Ontology Evaluation
CoMO was evaluated using complementary expert- and corpus-based approaches. Experts assessed semantic validity using Hootation[^1], which verbalizes ontology axioms as natural-language statements to facilitate expert review. In parallel, ontology coverage and semantic coherence were assessed using an automated evaluation pipeline previously developed for BSO-AD[^2] and applied to a large corpus of PubMed abstracts.

[^1]: Amith, M. et al. Expressing Biomedical Ontologies in Natural Language for Expert Evaluation. Stud. Health Technol. Inform. 245, 838–842 (2017). 
[^2]: Li, H. et al. BSO-AD: An Ontology for Representing and Harmonizing Behavioral Social Knowledge in ADRD. 2026.03.30.26349756 Preprint at https://doi.org/10.64898/2026.03.30.26349756 (2026). 

# Quick Start

## Set up
Create and activate a Python environment:

```bash
conda create -n como_env python=3.10
conda activate como_env
pip install -r requirements.txt
```
### LLM backends

| Model | Backend | Setup |
|---|---|---|
| GPT-5.5 | Mayo Clinic Azure OpenAI gateway (`--mayo`) | Create a `.env` file with `APIGEEX_CLIENT_ID` and `APIGEEX_SECRET_ID`, and set `ENV_FILE` to its path. This gateway is institution-internal. |
| Qwen3-235B-A22B-Instruct-2507 | Any OpenAI-compatible server, e.g. vLLM | Start the server before running Qwen steps (see below). |

Serving Qwen3 with vLLM:

```bash
vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507 --port 8000 --tensor-parallel-size 8
```

## Run
Run the scripts in order with `bash <script>`. GPT and Qwen steps can run
on different machines.

| Step | Script | Output |
|---|---|---|
| 0 | `step0_generate_150_evaluation_seed.sh` | `seed_concepts_150.json` |
| 1 | `step1_gpt_job_stage1.sh` | `l1_gpt.json` |
| 1 | `step1_qwen_job_stage1.sh` | `l1_qwen.json` |
| 1 | `step1_merge_gpt_qwen_stage1_results.sh` | `l1_assignment.json` |
| 2 | `step2_run_stage2.sh` | `proposed_skeleton.json`, `approved_skeleton.json` |
| 3 | `step3_run_stage3_gpt.sh` | `build_gpt.json` |
| 3 | `step3_run_stage3_qwen.sh` | `build_qwen.json` |
| 3 | `step3_run_stage3_gpt_qwen_check.sh` | `cross_*.json`, `l1/l2_review_sheet.csv` |

# Citation

# License
CoMO is licensed under the Creative Commons Attribution 4.0 International Public License(CC BY 4.0). Please see the License File for more information.
