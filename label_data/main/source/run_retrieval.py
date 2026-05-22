from retrieval_pipeline import run_retrieval

import numpy as np
import json

dataset = json.load(open("label_data/main/source/final/exp_high_aug/train.json"))

kb = [json.loads(line) for line in open("label_data/main/data/KB.jsonl", "r", encoding="utf-8")]

emb = np.load("label_data/main/data/kb_embeddings.npy")

results = run_retrieval(
    mode="test",
    dataset=dataset,
    kb_data=kb,
    kb_embeddings=emb,
    config={
        "use_query_expansion_llm": False,
        "use_rerank_llm": False,
        "max_candidates": 20,
        "max_top_k": 3
    }
)

print(results[0])