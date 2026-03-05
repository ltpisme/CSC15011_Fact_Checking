

import json
import numpy as np
from FlagEmbedding import BGEM3FlagModel


# 1. Khởi tạo mô hình
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

def build_kb_cache(json_path, output_npy_path):
    print(f"--- Bắt đầu xử lý file: {json_path} ---")
    with open(json_path, 'r', encoding='utf-8') as f:
        kb_data = json.load(f)
    
    kb_texts = [item['text'] for item in kb_data]
    
    # 2. Embedding toàn bộ (nên dùng batch_size lớn nếu có RAM tốt)
    print(f"Đang tính toán vector cho {len(kb_texts)} đoạn văn...")
    embeddings = model.encode(
        kb_texts, 
        batch_size=32, 
        max_length=8192 # Bạn có thể tăng lên 8192 nếu văn bản rất dài
    )['dense_vecs']
    
    # 3. Lưu xuống file .npy
    np.save(output_npy_path, embeddings)
    print(f"--- Đã lưu Cache thành công tại: {output_npy_path} ---")

if __name__ == "__main__":
    # Thay đường dẫn file của bạn vào đây
    build_kb_cache("../knowledge_base.json", "kb_embeddings.npy")