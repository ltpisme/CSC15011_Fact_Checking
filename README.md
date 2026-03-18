# Vietnamese Politic Factchecking

## Project Structure

```text
.
├── knowledge_base/
│   ├── KB.jsonl
│   ├── kb_embeddings.npy
│   └── Analysis.md
├── label_data/
│   ├── main/
│   │   ├── data/
│   │   │   ├── context/
│   │   │   │   └── context_claim_gen.jsonl
│   │   │   └── claim/
│   │   │       ├── claims_output.json
│   │   │       ├── claims_evaluate.json
│   │   │       └── failed_log.json
│   │   ├── dataset/
│   │   │   └── output_dataset.json
│   │   ├── source/
│   │   │   ├── configs.py
│   │   │   ├── build_cache_bge-m3.py
│   │   │   ├── claim_extract.py
│   │   │   ├── content_retrieval.py
│   │   │   └── main.py
│   │   └── final_label/
│   │       ├── EDA.ipynb
│   │       └── human_label_data.json
│   ├── pyproject.toml
│   └── uv.lock
└── link/
    ├── org/
    │   ├── org.json
    │   ├── org_raw.json
    │   ├── org_CT.json
    │   ├── org_CT.txt
    │   ├── org_TG.json
    │   └── org_TG.txt
    └── raw/
        └── combine/
```

## Directory Overview

- `knowledge_base/`  
  Contains processed data used for querying, analysis, and NLP tasks.

- `link/raw/`  
  Raw data crawled from multiple sources.

- `link/processed/`  
  Data that has been filtered, normalized, and is ready for use.

- `link/org/`  
  Raw data collected from the source [vnexpress.net](vnexpress.net).