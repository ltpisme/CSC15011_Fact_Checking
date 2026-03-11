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

import numpy as np


from FlagEmbedding import BGEM3FlagModel


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer



from configs import KB_PATH, KB_EMBED_PATH





def retrieve_top_k(claim, kb_data, k=3):
    """
        Sử dụng TF-IDF
    """
    texts = [item['text'] for item in kb_data]
    vectorizer = TfidfVectorizer().fit(texts + [claim])
    vectors = vectorizer.transform(texts)
    claim_vector = vectorizer.transform([claim])
    cosine_similarities = cosine_similarity(claim_vector, vectors).flatten()
    related_indices = cosine_similarities.argsort()[-k:][::-1]
    
    return [{"id": kb_data[idx]['id'], "text": kb_data[idx]['text'][:400], "source": kb_data[idx].get('source', 'Unknown')} for idx in related_indices]




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
    print("chạy hàm hybrid")
    # --- BƯỚC 1: BM25 (Sparse Retrieval) ---
    bm25 = _get_bm25_instance(kb_data)
    tokenized_query = ViTokenizer.tokenize(claim).lower().split()
    
    # Lấy điểm số BM25 cho tất cả các đoạn
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Lấy n index có điểm BM25 cao nhất
    candidate_indices = np.argsort(bm25_scores)[-n:][::-1]
    print(candidate_indices)
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



def compare_retrieval(claim, kb_data, kb_embeddings, k=3):
    # Lấy kết quả từ 2 phương pháp
    res_bge = retrieve_bge_m3(claim, kb_data, kb_embeddings, k=k)
    res_tfidf = retrieve_top_k(claim, kb_data, k=k)

  
    # Trích xuất ID hoặc Text để so sánh (giả sử mỗi item có 'id')
    set_bge = {item['id'] for item in res_bge}
    set_tfidf = {item['id'] for item in res_tfidf}

    # Tìm các phần tử chung
    common_ids = set_bge.intersection(set_tfidf)
    overlap_count = len(common_ids)

    print(f"\n[SO SÁNH KẾT QUẢ CHO CLAIM]: {claim}")
    print(f"{'='*60}")
    print(f"Số lượng bằng chứng trùng lặp: {overlap_count}/{k}")
    
    # In bảng so sánh tiêu đề/nguồn
    print(f"\n{'Hạng':<5} | {'BGE-M3 (Ngữ nghĩa)':<25} | {'TF-IDF (Từ khóa)':<25}")
    print(f"{'-'*60}")
    for i in range(k):
        source_bge = re.search(r'https?://([^/]+)', res_bge[i]['source']).group(1)
        source_tfidf = re.search(r'https?://([^/]+)', res_tfidf[i]['source']).group(1)
        
        # Đánh dấu nếu trùng
        mark = " (V)" if res_bge[i]['id'] in common_ids and res_bge[i]['id'] == res_tfidf[i]['id'] else ""
        
        print(f"{i+1:<5} | {source_bge:<25} | {source_tfidf:<25}{mark}")

    if overlap_count == 0:
        print("\n-> Nhận xét: Hai phương pháp đưa ra kết quả hoàn toàn khác nhau.")
    elif overlap_count == k:
        print("\n-> Nhận xét: Hai phương pháp thống nhất hoàn toàn về tập dữ liệu.")
    #
# --- CHẠY THỬ NGHIỆM ---
if __name__ == "__main__":
    test_claim = "Quan hệ VN - EU đã được nâng cấp lên doi tac Chiến lược toàn diện rồi bạn"
   
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb_data = json.load(f)
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    kb_embeddings = np.load(KB_EMBED_PATH)
    compare_retrieval(test_claim, kb_data, kb_embeddings)
   