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
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


def retrieve_top_k(claim, kb_data, k=3):
    texts = [item['text'] for item in kb_data]
    vectorizer = TfidfVectorizer().fit(texts + [claim])
    vectors = vectorizer.transform(texts)
    claim_vector = vectorizer.transform([claim])
    cosine_similarities = cosine_similarity(claim_vector, vectors).flatten()
    related_indices = cosine_similarities.argsort()[-k:][::-1]
    
    return [{"text": kb_data[idx]['text'][:400], "source": kb_data[idx].get('source', 'Unknown')} for idx in related_indices]


def retrieve_bge_m3(claim, kb_data, kb_embeddings, k=3):
    # 1. Encode duy nhất câu Claim (rất nhanh)
    claim_vec = model.encode([claim])['dense_vecs']
    
    # 2. Tính Cosine Similarity bằng ma trận (tốc độ ánh sáng)
    # Tích vô hướng giữa vector claim (1, dim) và ma trận KB (N, dim)
    similarities = np.dot(claim_vec, kb_embeddings.T).flatten()
    
    # 3. Lấy Top K indices
    top_k_indices = similarities.argsort()[-k:][::-1]
    
    # 4. Trả về nội dung đầy đủ (không cắt [:200] để LLM có đủ dữ liệu)
    results = []
    for idx in top_k_indices:
        results.append({
            "text": kb_data[idx]['text'],
            "source": kb_data[idx].get('source', 'Unknown'),
            "score": float(similarities[idx])
        })
    return results

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
    kb_path = "../knowledge_base.json"
    with open(kb_path, "r", encoding="utf-8") as f:
        kb_data = json.load(f)
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    kb_embeddings = np.load("kb_embeddings.npy")
    compare_retrieval(test_claim, kb_data, kb_embeddings)
   