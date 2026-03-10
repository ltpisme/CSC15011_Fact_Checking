"""
Module: Evidence Retrieval & Comparison
Description: Truy xuất các đoạn văn bản (Evidence) liên quan tới Luận điểm (Claim).
             Hỗ trợ so sánh hiệu năng giữa hai phương pháp tiếp cận:
             
    - BGE-M3 (Dense Retrieval): Sử dụng Embedding đa ngôn ngữ để tìm kiếm ngữ nghĩa, 
      tối ưu hóa qua file cache .npy.
    - TF-IDF (Sparse Retrieval): Sử dụng thuật toán truyền thống dựa trên tần suất 
      từ khóa để đảm bảo độ chính xác của các thực thể (tên riêng, ngày tháng).

Note: Kết quả so sánh giữa hai mô hình giúp tinh chỉnh chiến lược Retrieval 
      trước khi đưa vào giai đoạn dán nhãn LLM.
"""


import json
import re

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from configs import KB_PATH, KB_EMBED_PATH

_bge_model: BGEM3FlagModel | None = None


def _get_bge_model() -> BGEM3FlagModel:
    """Lazy-load BGE-M3 model as a module-level singleton."""
    global _bge_model
    if _bge_model is None:
        _bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return _bge_model






def retrieve_top_k(claim, kb_data, k=3):
    texts = [item['text'] for item in kb_data]
    vectorizer = TfidfVectorizer().fit(texts + [claim])
    vectors = vectorizer.transform(texts)
    claim_vector = vectorizer.transform([claim])
    cosine_similarities = cosine_similarity(claim_vector, vectors).flatten()
    related_indices = cosine_similarities.argsort()[-k:][::-1]
    
    return [{"id": kb_data[idx]['id'], "text": kb_data[idx]['text'][:400], "source": kb_data[idx].get('source', 'Unknown')} for idx in related_indices]


def retrieve_bge_m3(claim: str, kb_data: list[dict], kb_embeddings: np.ndarray, k: int = 3) -> list[dict]:
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
   