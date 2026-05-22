"""
Retrieval-only pipeline for fact-checking datasets.

Designed for:
- Kaggle notebooks
- Local notebooks
- Standalone Python runs

This file ONLY performs retrieval.

It DOES NOT:
- generate labels
- vote labels
- verify claims
- perform final fact-checking

Main idea:
TRAIN:
    - retrieve from sample contexts only
    - simulate evidence selection

TEST:
    - retrieve from KB using claim only
    - output evidence for downstream classifier

Downstream ML/DL classifier will later use:
    claim + retrieved_evidence -> predict label
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import numpy as np
from dataclasses import dataclass
from typing import Any, Literal
from dotenv import load_dotenv

load_dotenv()

Mode = Literal["train", "test"]

OPENROUTER_DEFAULT_URL = "https://openrouter.ai/api/v1/chat/completions"


# ============================================================
# CONFIG
# ============================================================

@dataclass(slots=True)
class RetrievalConfig:
    # ========================================================
    # Retrieval size
    # ========================================================

    min_top_k: int = 1
    max_top_k: int = 3

    # If True:
    # always return exactly max_top_k contexts
    #
    # If False:
    # return random number in [min_top_k, max_top_k]
    #
    # Examples:
    #   min=1 max=3 strict=False
    #       -> may return 1/2/3 contexts
    #
    #   min=3 max=3 strict=True
    #       -> always return 3 contexts
    #
    strict_top_k: bool = False

    # ========================================================
    # BM25
    # ========================================================

    bm25_alpha: float = 0.0
    max_candidates: int = 30

    # ========================================================
    # Dense rerank
    # ========================================================

    rerank_threshold: float | None = None

    bge_model_name: str = "BAAI/bge-m3"
    use_fp16: bool = True

    # ============================================================
    # LLM FLAGS
    # ============================================================

    use_query_expansion_llm: bool = True
    use_rerank_llm: bool = True

    # ============================================================
    # LLM
    # ============================================================

    # Query expansion model
    #
    # Recommended:
    #   qwen/qwen3-8b:free
    #
    # Why:
    #   - strong Vietnamese multilingual support
    #   - good semantic rewrite
    #   - fast and cheap
    #   - suitable for query expansion
    #
    query_expansion_model: str = "qwen/qwen3-8b:free"

    # LLM rerank model
    #
    # Recommended:
    #   google/gemma-3-27b-it:free
    #
    # Why:
    #   - stable reranking
    #   - good instruction following
    #   - conservative relevance scoring
    #   - less over-reasoning
    #
    rerank_model: str = "google/gemma-3-27b-it:free"

    openrouter_base_url: str = OPENROUTER_DEFAULT_URL

    llm_timeout: int = 60
    max_llm_retries: int = 3
    llm_backoff_seconds: float = 1.5

    @classmethod
    def from_mapping(
        cls,
        config: dict[str, Any] | "RetrievalConfig" | None,
    ) -> "RetrievalConfig":

        if config is None:
            return cls()

        if isinstance(config, cls):
            return config

        defaults = cls()

        values = {
            "min_top_k": config.get(
                "min_top_k",
                config.get("MIN_TOP_K", defaults.min_top_k),
            ),
            "max_top_k": config.get(
                "max_top_k",
                config.get("MAX_TOP_K", defaults.max_top_k),
            ),
            "strict_top_k": config.get(
                "strict_top_k",
                config.get("STRICT_TOP_K", defaults.strict_top_k),
            ),
            "bm25_alpha": config.get(
                "bm25_alpha",
                config.get("BM25_ALPHA", defaults.bm25_alpha),
            ),
            "max_candidates": config.get(
                "max_candidates",
                config.get("MAX_CANDIDATES", defaults.max_candidates),
            ),
            "rerank_threshold": config.get(
                "rerank_threshold",
                config.get(
                    "RERANK_THRESHOLD",
                    defaults.rerank_threshold,
                ),
            ),
            "bge_model_name": config.get(
                "bge_model_name",
                config.get(
                    "BGE_MODEL_NAME",
                    defaults.bge_model_name,
                ),
            ),
            "use_fp16": config.get(
                "use_fp16",
                config.get(
                    "USE_FP16",
                    defaults.use_fp16,
                ),
            ),
            "use_query_expansion_llm": config.get(
                "use_query_expansion_llm",
                config.get(
                    "USE_QUERY_EXPANSION_LLM",
                    defaults.use_query_expansion_llm,
                ),
            ),

            "use_rerank_llm": config.get(
                "use_rerank_llm",
                config.get(
                    "USE_RERANK_LLM",
                    defaults.use_rerank_llm,
                ),
            ),
            "query_expansion_model": config.get(
                "query_expansion_model",
                config.get(
                    "QUERY_EXPANSION_MODEL",
                    defaults.query_expansion_model,
                ),
            ),

            "rerank_model": config.get(
                "rerank_model",
                config.get(
                    "RERANK_MODEL",
                    defaults.rerank_model,
                ),
            ),
            "openrouter_base_url": config.get(
                "openrouter_base_url",
                config.get(
                    "OPENROUTER_BASE_URL",
                    defaults.openrouter_base_url,
                ),
            ),
            "llm_timeout": config.get(
                "llm_timeout",
                config.get(
                    "LLM_TIMEOUT",
                    defaults.llm_timeout,
                ),
            ),
            "max_llm_retries": config.get(
                "max_llm_retries",
                config.get(
                    "MAX_LLM_RETRIES",
                    defaults.max_llm_retries,
                ),
            ),
            "llm_backoff_seconds": config.get(
                "llm_backoff_seconds",
                config.get(
                    "LLM_BACKOFF_SECONDS",
                    defaults.llm_backoff_seconds,
                ),
            ),
        }

        return cls(**values)

    @property
    def resolved_openrouter_api_key(self) -> str | None:
        return (
            os.getenv("OPENROUTER_API_KEY")
        )


# ============================================================
# GLOBALS
# ============================================================

_bge_model = None


# ============================================================
# UTILITIES
# ============================================================

def _tokenize(text: str) -> list[str]:

    try:
        from pyvi import ViTokenizer

        return ViTokenizer.tokenize(
            (text or "").lower()
        ).split()

    except ImportError:
        return re.findall(
            r"\w+",
            (text or "").lower(),
            flags=re.UNICODE,
        )


def _normalize_text(text: str) -> str:

    return re.sub(
        r"\s+",
        " ",
        (text or "").strip().lower(),
    )


def _normalize_scores(scores: list[float]) -> list[float]:

    if not scores:
        return []

    low = min(scores)
    high = max(scores)

    if low == high:
        return [0.0 for _ in scores]

    return [
        (score - low) / (high - low)
        for score in scores
    ]


def _choose_top_k(config: RetrievalConfig) -> int:

    if config.strict_top_k:
        return config.max_top_k

    return random.randint(
        config.min_top_k,
        config.max_top_k,
    )


def _gold_contexts(sample: dict[str, Any]) -> list[str]:
    """
    Support multiple dataset formats.

    Supported formats:

    1.
    {
        "contexts": [
            "...",
            "...",
            "..."
        ]
    }

    2.
    {
        "context_1": "...",
        "context_2": "...",
        "context_3": "..."
    }
    """

    # ============================================
    # FORMAT 1:
    # contexts: []
    # ============================================

    contexts = sample.get("contexts")

    if isinstance(contexts, list):

        cleaned = []

        for ctx in contexts:

            if (
                isinstance(ctx, str)
                and ctx.strip()
            ):
                cleaned.append(ctx.strip())

        return cleaned

    # ============================================
    # FORMAT 2:
    # context_1 / context_2 / context_3
    # ============================================

    legacy_contexts = [
        sample.get("context_1", ""),
        sample.get("context_2", ""),
        sample.get("context_3", ""),
    ]

    cleaned = []

    for ctx in legacy_contexts:

        if (
            isinstance(ctx, str)
            and ctx.strip()
        ):
            cleaned.append(ctx.strip())

    return cleaned


def _as_corpus(
    contexts: list[str],
    source: str = "oracle",
) -> list[dict[str, Any]]:

    return [
        {
            "id": f"{source}_{idx}",
            "text": text,
            "source": source,
        }
        for idx, text in enumerate(contexts, start=1)
    ]


# ============================================================
# OPENROUTER
# ============================================================

def _extract_json_object(text: str) -> Any | None:

    text = (text or "").strip()

    if not text:
        return None

    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL,
    )

    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)

    except Exception:
        pass

    match = re.search(
        r"(\{.*\}|\[.*\])",
        text,
        flags=re.DOTALL,
    )

    if not match:
        return None

    try:
        return json.loads(match.group(1))

    except Exception:
        return None


def _call_openrouter(
    prompt: str,
    model_name: str,
    config: RetrievalConfig,
    system_prompt: str = "You are a retrieval assistant.",
) -> str | None:

    api_key = config.resolved_openrouter_api_key

    if not api_key:
        return None

    try:
        import requests

    except ImportError:
        return None

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(config.max_llm_retries):

        try:
            response = requests.post(
                config.openrouter_base_url,
                headers=headers,
                json=payload,
                timeout=config.llm_timeout,
            )

            response.raise_for_status()

            data = response.json()

            return data["choices"][0]["message"]["content"]

        except Exception:

            if attempt == config.max_llm_retries - 1:
                return None

            time.sleep(
                config.llm_backoff_seconds * (2**attempt)
            )

    return None


# ============================================================
# LLM QUERY EXPANSION
# ============================================================

def _expand_query_with_llm(
    claim: str,
    config: RetrievalConfig,
) -> str:

    if not config.use_query_expansion_llm:
        return claim

    prompt = f"""
Expand the Vietnamese fact-checking claim into a better retrieval query.

Rules:
- Preserve original meaning
- Add aliases and retrieval keywords
- Do NOT fact-check
- Do NOT predict labels
- Output ONLY JSON

Format:
{{"expanded_query":"..."}}

Claim:
{claim}
""".strip()

    content = _call_openrouter(
        prompt=prompt,
        model_name=config.query_expansion_model,
        config=config,
    )

    parsed = _extract_json_object(content or "")

    if (
        isinstance(parsed, dict)
        and isinstance(parsed.get("expanded_query"), str)
    ):
        expanded = parsed["expanded_query"].strip()

        if expanded:
            return expanded

    return claim


# ============================================================
# LLM RERANK
# ============================================================

def _rerank_with_llm(
    claim: str,
    retrieved_evidence: list[dict[str, Any]],
    config: RetrievalConfig,
) -> list[dict[str, Any]]:

    if (
        not config.use_rerank_llm
        or not retrieved_evidence
    ):
        return retrieved_evidence

    context_lines = []

    for idx, item in enumerate(retrieved_evidence, start=1):

        text = item["text"].replace("\n", " ")

        context_lines.append(
            f"[{idx}] {text[:1500]}"
        )

    prompt = f"""
Rerank contexts for retrieval relevance.

Rules:
- Focus ONLY on retrieval usefulness
- Do NOT fact-check
- Do NOT predict labels
- Return ALL ids best-first
- Output ONLY JSON

Format:
{{
    "ranking":[
        {{"id":1,"score":0.95}}
    ]
}}

Claim:
{claim}

Contexts:
{chr(10).join(context_lines)}
""".strip()

    content = _call_openrouter(
        prompt=prompt,
        model_name=config.rerank_model,
        config=config,
    )

    parsed = _extract_json_object(content or "")

    if (
        not isinstance(parsed, dict)
        or not isinstance(parsed.get("ranking"), list)
    ):
        return retrieved_evidence

    original = {
        idx: item
        for idx, item in enumerate(
            retrieved_evidence,
            start=1,
        )
    }

    reranked = []
    used = set()

    for entry in parsed["ranking"]:

        if not isinstance(entry, dict):
            continue

        idx = entry.get("id")

        if (
            not isinstance(idx, int)
            or idx not in original
        ):
            continue

        item = dict(original[idx])

        if isinstance(entry.get("score"), (int, float)):
            item["score"] = float(entry["score"])

        reranked.append(item)

        used.add(idx)

    for idx, item in original.items():

        if idx not in used:
            reranked.append(item)

    return reranked


# ============================================================
# BGE
# ============================================================

def _get_bge_model(config: RetrievalConfig):

    global _bge_model

    if _bge_model is None:

        from FlagEmbedding import BGEM3FlagModel

        _bge_model = BGEM3FlagModel(
            config.bge_model_name,
            use_fp16=config.use_fp16,
        )

    return _bge_model


# ============================================================
# RETRIEVER
# ============================================================

class HybridRetriever:

    def __init__(
        self,
        corpus: list[dict[str, Any]],
        embeddings: Any | None,
        config: RetrievalConfig,
    ):

        self.corpus = corpus
        self.embeddings = embeddings
        self.config = config
        self._bm25 = None

    def retrieve(self, claim: str):

        query = _expand_query_with_llm(
            claim,
            self.config,
        )

        candidates = self._bm25_candidates(query)

        if not candidates:
            return []

        if self.embeddings is None:

            evidence = self._format_results(
                candidates,
                score_key="score_sparse_norm",
            )

        else:

            evidence = self._dense_rerank(
                query,
                candidates,
            )

        evidence = _rerank_with_llm(
            claim,
            evidence,
            self.config,
        )

        return self._finalize(evidence)

    def _bm25_candidates(
        self,
        query: str,
    ):

        tokenized_query = _tokenize(query)

        if self._bm25 is None:

            tokenized_corpus = [
                _tokenize(item["text"])
                for item in self.corpus
            ]

            try:
                from rank_bm25 import BM25Okapi

                self._bm25 = BM25Okapi(
                    tokenized_corpus
                )

            except ImportError:
                self._bm25 = tokenized_corpus

        if hasattr(self._bm25, "get_scores"):

            scores = [
                float(score)
                for score in self._bm25.get_scores(
                    tokenized_query
                )
            ]

        else:

            query_terms = set(tokenized_query)

            scores = [
                float(
                    len(query_terms & set(tokens))
                    / max(len(query_terms), 1)
                )
                for tokens in self._bm25
            ]

        normalized_scores = _normalize_scores(scores)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )

        thresholded = [
            idx
            for idx in ranked_indices
            if normalized_scores[idx]
            >= self.config.bm25_alpha
        ]

        selected = (
            thresholded or ranked_indices
        )[: min(
            self.config.max_candidates,
            len(self.corpus),
        )]

        return [
            {
                **self.corpus[idx],
                "corpus_index": idx,
                "score_sparse": float(scores[idx]),
                "score_sparse_norm": float(
                    normalized_scores[idx]
                ),
            }
            for idx in selected
        ]

    def _dense_rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ):

        if np is None:
            raise ImportError(
                "numpy is required for dense rerank"
            )

        model = _get_bge_model(self.config)

        query_vec = model.encode([query])[
            "dense_vecs"
        ]

        candidate_indices = np.asarray(
            [
                item["corpus_index"]
                for item in candidates
            ],
            dtype=int,
        )

        candidate_embeddings = self.embeddings[
            candidate_indices
        ]

        scores = np.dot(
            query_vec,
            candidate_embeddings.T,
        ).flatten()

        ranked_positions = np.argsort(scores)[::-1]

        ranked = []

        for position in ranked_positions:

            dense_score = float(scores[position])

            if (
                self.config.rerank_threshold
                is not None
                and dense_score
                < self.config.rerank_threshold
            ):
                continue

            item = dict(candidates[position])

            item["score"] = dense_score

            ranked.append(item)

        return ranked

    def _format_results(
        self,
        ranked: list[dict[str, Any]],
        score_key: str,
    ):

        evidence = []

        for item in ranked:

            evidence.append(
                {
                    "text": item["text"],
                    "score": float(
                        item.get(
                            score_key,
                            item.get("score", 0.0),
                        )
                    ),
                }
            )

        return evidence

    def _finalize(
        self,
        evidence: list[dict[str, Any]],
    ):

        top_k = _choose_top_k(self.config)

        selected = evidence[:top_k]

        final = []

        for rank, item in enumerate(
            selected,
            start=1,
        ):

            final.append(
                {
                    "text": item["text"],
                    "score": float(
                        item.get("score", 0.0)
                    ),
                    "rank": rank,
                }
            )

        return final


# ============================================================
# EVALUATION
# ============================================================

def _evaluate_retrieval(
    retrieved_evidence: list[dict[str, Any]],
    gold_contexts: list[str],
):

    gold = {
        _normalize_text(x)
        for x in gold_contexts
    }

    correct = []

    for idx, item in enumerate(
        retrieved_evidence,
        start=1,
    ):

        if _normalize_text(item["text"]) in gold:
            correct.append(idx)

    return {
        "recall": 1.0 if correct else 0.0,
        "mrr": (
            1.0 / correct[0]
            if correct
            else 0.0
        ),
        "precision": (
            len(correct)
            / max(len(retrieved_evidence), 1)
        ),
    }


# ============================================================
# TRAIN
# ============================================================

def _run_train(
    dataset: list[dict[str, Any]],
    config: RetrievalConfig,
):

    results = []

    for sample in dataset:

        claim = sample.get("claim", "")

        gold_contexts = _gold_contexts(sample)

        corpus = _as_corpus(gold_contexts)

        retriever = HybridRetriever(
            corpus=corpus,
            embeddings=None,
            config=config,
        )

        retrieved = retriever.retrieve(claim)

        metrics = _evaluate_retrieval(
            retrieved,
            gold_contexts,
        )

        results.append(
            {
                "claim": claim,
                "label": sample.get("label"),
                "retrieved_evidence": retrieved,
                "metrics": metrics,
            }
        )

    return results


# ============================================================
# TEST
# ============================================================

def _run_test(
    dataset: list[dict[str, Any]],
    kb_data: list[dict[str, Any]],
    kb_embeddings: Any,
    config: RetrievalConfig,
):

    if np is None:
        raise ImportError(
            "numpy required for test mode"
        )

    if kb_embeddings is None:
        raise ValueError(
            "kb_embeddings required for test mode"
        )

    print("KB size:", len(kb_data))
    print("Embeddings size:", len(kb_embeddings))

    if len(kb_data) != len(kb_embeddings):

        raise ValueError(
            "KB size and embedding size mismatch"
        )

    retriever = HybridRetriever(
        corpus=kb_data,
        embeddings=kb_embeddings,
        config=config,
    )

    results = []

    for sample in dataset:

        claim = sample.get("claim", "")

        retrieved = retriever.retrieve(claim)

        results.append(
            {
                "claim": claim,
                "label": sample.get("label"),
                "retrieved_evidence": retrieved,
            }
        )

    return results


# ============================================================
# PUBLIC API
# ============================================================

def run_retrieval(
    mode: Mode,
    dataset: list[dict[str, Any]],
    kb_data: list[dict[str, Any]] | None = None,
    kb_embeddings: Any | None = None,
    config: dict[str, Any]
    | RetrievalConfig
    | None = None,
):

    config = RetrievalConfig.from_mapping(config)

    mode = mode.lower()

    if mode == "train":

        return _run_train(
            dataset,
            config,
        )

    if mode == "test":

        return _run_test(
            dataset,
            kb_data or [],
            kb_embeddings,
            config,
        )

    raise ValueError(
        "mode must be 'train' or 'test'"
    )