# ENA Data Format Tool

This repository contains a local tooling pipeline for preparing textual data for Epistemic Network Analysis (ENA) and concept-based semantic analysis.

It supports:
- Segmenting raw text into analysable units
- Defining conceptual schemas in a single source of truth (`concepts.yaml`)
- Generating concept paradigm sentences using an LLM API
- Producing clean, reproducible CSV formats for downstream analysis (ENA, similarity coding, distillation, etc.)

---

## 📦 Repository Structure

```text
ena_data_format/
├── ena_tool.py
├── schemas/
│   └── concepts.yaml        # Concept ontology / schema
├── data_in/
│   └── raw.csv              # Example raw input data
├── data_out/
│   ├── units.csv            # Segmented units (output of segment_csv)
│   └── paradigms_zh.csv     # Generated paradigm sentences (output of paradigms)
└── README.md
