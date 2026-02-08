# Vietnamese Politic Factchecking

## Project Structure

```text
.
├── knowledge_base
├── link
│   ├── org
│   │   ├── org_CT.json
│   │   ├── org.json
│   │   ├── org_raw.json
│   │   └── org_TG.json
│   ├── processed
│   └── raw
│       ├── Bao
│       ├── combine
│       ├── Kien
│       └── Van
├── README.md
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