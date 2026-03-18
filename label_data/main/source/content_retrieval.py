"""
Module: Evidence Retrieval & Comparison
Description: Truy xuất các đoạn văn bản (Evidence) liên quan tới Luận điểm (Claim).
             Cung cấp ba chiến lược retrieval độc lập, cho phép so sánh và lựa chọn
             phương pháp phù hợp nhất trước khi đưa vào pipeline dán nhãn.

Các phương pháp được hỗ trợ:
    - TF-IDF (Sparse Retrieval):
        Tìm kiếm dựa trên tần suất từ khóa (sklearn). Không cần model hay cache,
        thích hợp để kiểm thử nhanh và đảm bảo độ chính xác trên thực thể
        (tên riêng, số liệu, ngày tháng).

    - BGE-M3 (Dense Retrieval):
        Sử dụng model embedding đa ngôn ngữ `BAAI/bge-m3` để tìm kiếm theo ngữ
        nghĩa. Vector KB được pre-compute và cache ra file `.npy` để tăng tốc độ.
        Model được lazy-load dưới dạng singleton, chỉ khởi tạo một lần duy nhất.

    - Hybrid BM25 + BGE-M3 (Two-stage Retrieval):
        Kết hợp Sparse và Dense retrieval theo pipeline 2 lớp:
          1. BM25 (ViTokenizer) lọc nhanh n ứng viên có từ khóa/con số trùng khớp.
          2. BGE-M3 rerank k đoạn văn có ngữ nghĩa khớp nhất từ tập n ứng viên đó.
        Vừa chính xác về thực thể, vừa bắt được ngữ nghĩa sâu hơn so với TF-IDF
        hay BGE-M3 đơn lẻ.

Note: `compare_retrieval()` xuất bảng so sánh song song BGE-M3 vs TF-IDF, dùng
      để đánh giá và tinh chỉnh chiến lược trước khi chốt pipeline chính thức.
"""


import json
import re
import random

import numpy as np


from FlagEmbedding import BGEM3FlagModel


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer



from configs import KB_PATH, KB_EMBED_PATH, BASE_DIR

KB_VNEXPRESS_PATH = BASE_DIR / "label_data/main/data/KB_vnexpress.jsonl"





def retrieve_tfidf(claim, kb_data, k=3):
    """
        Sử dụng TF-IDF
    """
    texts = [item['text'] for item in kb_data]
    vectorizer = TfidfVectorizer().fit(texts + [claim])
    vectors = vectorizer.transform(texts)
    claim_vector = vectorizer.transform([claim])
    cosine_similarities = cosine_similarity(claim_vector, vectors).flatten()
    related_indices = cosine_similarities.argsort()[-k:][::-1]
    
    return [
        {
            "id": kb_data[idx]["id"],
            "text": kb_data[idx]["text"],
            "source": kb_data[idx].get("source", "Unknown"),
            "score": float(cosine_similarities[idx]),
        }
        for idx in related_indices
    ]




_bm25_instance = None
_bge_model: BGEM3FlagModel | None = None


def _get_bge_model() -> BGEM3FlagModel:
    """Lazy-load BGE-M3 model as a module-level singleton."""
    global _bge_model
    if _bge_model is None:
        _bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return _bge_model


def _get_bm25_instance(kb_data):
    global _bm25_instance
    if _bm25_instance is None:
        # Tokenize toàn bộ KB một lần duy nhất
        tokenized_corpus = [ViTokenizer.tokenize(doc['text']).lower().split() for doc in kb_data]
        _bm25_instance = BM25Okapi(tokenized_corpus)
    return _bm25_instance

def retrieve_bge_m3(claim: str, kb_data: list[dict], kb_embeddings: np.ndarray, k: int = 3) -> list[dict]:
    """
        Sử dụng BGE-M3
    """
    model = _get_bge_model()
    claim_vec = model.encode([claim])["dense_vecs"]
    
    # Tích vô hướng giữa vector claim (1, dim) và ma trận KB (N, dim)
    similarities = np.dot(claim_vec, kb_embeddings.T).flatten()
    top_k_indices = similarities.argsort()[-k:][::-1]

    return [
        {
            "id": kb_data[idx]["id"],
            "text": kb_data[idx]["text"],
            "source": kb_data[idx].get("source", "Unknown"),
            "score": float(similarities[idx]),
        }
        for idx in top_k_indices
    ]


def retrieve_hybrid_bge(claim: str, kb_data: list[dict], kb_embeddings: np.ndarray, n: int = 30, k: int = 3):
    """
    Pipeline 2 lớp:
    1. BM25 lọc ra n đoạn văn chứa từ khóa/con số giống nhất.
    2. BGE-M3 chọn ra k đoạn văn có ngữ nghĩa khớp nhất từ n đoạn đó.
    """
    # print("chạy hàm hybrid")
    # --- BƯỚC 1: BM25 (Sparse Retrieval) ---
    bm25 = _get_bm25_instance(kb_data)
    tokenized_query = ViTokenizer.tokenize(claim).lower().split()
    
    # Lấy điểm số BM25 cho tất cả các đoạn
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Lấy n index có điểm BM25 cao nhất
    candidate_indices = np.argsort(bm25_scores)[-n:][::-1]
    # print(candidate_indices)
    # --- BƯỚC 2: BGE-M3 (Dense Reranking) ---
    model = _get_bge_model()
    claim_vec = model.encode([claim])["dense_vecs"] # (1, dim)
    
    # Chỉ lấy các vector embedding của n ứng viên đã lọc
    candidate_embeddings = kb_embeddings[candidate_indices] # (n, dim)
    
    # Tính độ tương đồng trên tập ứng viên (n nhỏ nên cực nhanh)
    similarities = np.dot(claim_vec, candidate_embeddings.T).flatten()
    
    # Lấy k index tốt nhất từ n ứng viên
    rel_top_k_indices = similarities.argsort()[-k:][::-1]
    final_indices = [candidate_indices[i] for i in rel_top_k_indices]

    return [
        {
            "id": kb_data[idx]["id"],
            "text": kb_data[idx]["text"],
            "score_dense": float(similarities[i]),
            "score_sparse": float(bm25_scores[idx]),
            "source": kb_data[idx].get("source", "Unknown")
        }
        for i, idx in enumerate(final_indices)
    ]



def compare_retrieval(kb_data: list[dict], kb_embeddings: np.ndarray, n_claims: int = 10, k: int = 3):
    """So sánh 3 phương pháp retrieval (TF-IDF, BGE-M3, Hybrid) trên n_claims được lấy ngẫu nhiên từ KB_vnexpress."""
    with open(KB_VNEXPRESS_PATH, "r", encoding="utf-8") as f:
        vnexpress_data = [json.loads(line) for line in f if line.strip()]

    sample_claims = random.sample(vnexpress_data, min(n_claims, len(vnexpress_data)))

    total_overlap_tfidf_bge = 0
    total_overlap_tfidf_hybrid = 0
    total_overlap_bge_hybrid = 0

    for idx, item in enumerate(sample_claims):
        claim = item["text"][:200]  # Dùng 200 ký tự đầu làm claim
        print(f"\n{'='*80}")
        print(f"[CLAIM {idx+1}/{n_claims}]: {claim[:100]}...")
        print(f"{'='*80}")

        res_tfidf  = retrieve_tfidf(claim, kb_data, k=k)
        res_bge    = retrieve_bge_m3(claim, kb_data, kb_embeddings, k=k)
        res_hybrid = retrieve_hybrid_bge(claim, kb_data, kb_embeddings, k=k)

        ids_tfidf  = {r["id"] for r in res_tfidf}
        ids_bge    = {r["id"] for r in res_bge}
        ids_hybrid = {r["id"] for r in res_hybrid}

        overlap_tfidf_bge    = len(ids_tfidf & ids_bge)
        overlap_tfidf_hybrid = len(ids_tfidf & ids_hybrid)
        overlap_bge_hybrid   = len(ids_bge & ids_hybrid)

        total_overlap_tfidf_bge    += overlap_tfidf_bge
        total_overlap_tfidf_hybrid += overlap_tfidf_hybrid
        total_overlap_bge_hybrid   += overlap_bge_hybrid

        # Bảng kết quả
        print(f"\n{'Hạng':<5} | {'TF-IDF (score)':<40} | {'BGE-M3 (score)':<40} | {'Hybrid (dense|sparse)':<45}")
        print(f"{'-'*140}")
        for i in range(k):
            src_tfidf  = res_tfidf[i]["source"][:35]
            src_bge    = res_bge[i]["source"][:35]
            src_hybrid = res_hybrid[i]["source"][:35]
            sc_tfidf   = f"{res_tfidf[i]['score']:.4f}"
            sc_bge     = f"{res_bge[i]['score']:.4f}"
            sc_hybrid  = f"{res_hybrid[i]['score_dense']:.4f}|{res_hybrid[i]['score_sparse']:.4f}"
            print(f"{i+1:<5} | {src_tfidf+' ('+sc_tfidf+')':<40} | {src_bge+' ('+sc_bge+')':<40} | {src_hybrid+' ('+sc_hybrid+')':<45}")

        print(f"\n  Overlap TF-IDF ∩ BGE-M3   : {overlap_tfidf_bge}/{k}")
        print(f"  Overlap TF-IDF ∩ Hybrid   : {overlap_tfidf_hybrid}/{k}")
        print(f"  Overlap BGE-M3 ∩ Hybrid   : {overlap_bge_hybrid}/{k}")

    print(f"\n{'='*80}")
    print(f"TỔNG KẾT ({n_claims} claims, top-{k}):")
    print(f"  Trung bình overlap TF-IDF ∩ BGE-M3 : {total_overlap_tfidf_bge/n_claims:.2f}/{k}")
    print(f"  Trung bình overlap TF-IDF ∩ Hybrid : {total_overlap_tfidf_hybrid/n_claims:.2f}/{k}")
    print(f"  Trung bình overlap BGE-M3 ∩ Hybrid : {total_overlap_bge_hybrid/n_claims:.2f}/{k}")
    print(f"{'='*80}")


def export_retrieval_results(kb_data: list[dict], kb_embeddings: np.ndarray, generate_claims_fn, n_contexts: int = 10, k: int = 3, output_dir: str = "."):
    """Xuất kết quả retrieval của 3 phương pháp ra 3 file JSON riêng biệt.
    
    Args:
        generate_claims_fn: hàm generate_claims từ main.py, truyền vào để tránh circular import.
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(KB_VNEXPRESS_PATH, "r", encoding="utf-8") as f:
        vnexpress_data = [json.loads(line) for line in f if line.strip()]

    seed_contexts = random.sample(vnexpress_data, min(n_contexts, len(vnexpress_data)))

    tfidf_results  = []
    bge_results    = []
    hybrid_results = []

    claim_count = 0
    for ctx_idx, item in enumerate(seed_contexts):
        seed_text = item["text"]
        print(f"[Context {ctx_idx+1}/{n_contexts}] Đang sinh claim từ: {seed_text[:80]}...")

        claims = generate_claims_fn(seed_text)
        if not claims:
            print(f"  -> Không sinh được claim, bỏ qua.")
            continue

        for claim_item in claims:
            claim_text = claim_item.get("claim", "")
            if not claim_text:
                continue

            claim_count += 1
            print(f"  [{claim_count}] Retrieval cho: {claim_text[:80]}...")

            res_tfidf  = retrieve_tfidf(claim_text, kb_data, k=k)
            res_bge    = retrieve_bge_m3(claim_text, kb_data, kb_embeddings, k=k)
            res_hybrid = retrieve_hybrid_bge(claim_text, kb_data, kb_embeddings, k=k)

            entry = {
                "claim_id": f"ctx{ctx_idx+1}_claim{claim_count}",
                "claim": claim_text,
                "label": claim_item.get("label", ""),
                "seed_context_id": item.get("id", f"ctx_{ctx_idx+1}"),
            }

            tfidf_results.append({**entry, "evidences": res_tfidf})
            bge_results.append({**entry, "evidences": res_bge})
            hybrid_results.append({**entry, "evidences": res_hybrid})

    files = {
        "tfidf_retrieval.json": tfidf_results,
        "bge-m3_retrieval.json": bge_results,
        "hybrid_retrieval.json": hybrid_results,
    }

    for filename, data in files.items():
        path = output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Đã xuất: {path}")


# --- CHẠY THỬ NGHIỆM ---
if __name__ == "__main__":
    from main import generate_claims

    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb_data = [json.loads(line) for line in f if line.strip()]
    kb_embeddings = np.load(KB_EMBED_PATH)
    export_retrieval_results(kb_data, kb_embeddings, generate_claims_fn=generate_claims, n_contexts=10, k=3, output_dir=".")
   