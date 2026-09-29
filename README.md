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

# A Quick Start

# Citation

# License
CoMO is licensed under the Creative Commons Attribution 4.0 International Public License(CC BY 4.0). Please see the License File for more information.
