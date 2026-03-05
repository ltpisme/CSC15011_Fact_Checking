import json
import os
from trafilatura import fetch_url, extract
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Giả sử hàm process_article_to_json đã được định nghĩa như trước
def process_article_to_json(url, article_id):
    try:
        downloaded = fetch_url(url)
        if not downloaded:
            return []
        
        raw_text = extract(downloaded)
        if not raw_text:
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=50,
            # separators=["\n\n", "\n", ". ", " "]
        )
        chunks = text_splitter.split_text(raw_text)

        entries = []
        for i, chunk in enumerate(chunks):
            entries.append({
                "id": f"{article_id}_chunk_{i:03d}",
                "source": url,
                "text": chunk,
                "metadata": {
                    "topic": "Finance/Policy",
                    "date": "2026-01-29"
                }
            })
        return entries
    except Exception as e:
        print(f"Lỗi khi xử lý {url}: {e}")
        return []

def main():
    input_file = "newspaper_url.txt"
    output_file = "knowledge_base.json"
    
    # 1. Kiểm tra file input
    if not os.path.exists(input_file):
        print(f"❌ Không tìm thấy file {input_file}")
        return

    # 2. Đọc danh sách URL từ file
    with open(input_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    all_data = []
    
    # Nếu file output đã tồn tại, đọc dữ liệu cũ để append tiếp (tránh ghi đè mất dữ liệu cũ)
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                all_data = json.load(f)
        except:
            all_data = []

    # 3. Lặp qua từng URL để xử lý
    for idx, url in enumerate(urls):
        doc_id = f"DOC_{len(all_data) + 1:03d}" # Đánh số tiếp nối dữ liệu cũ
        print(f"🚀 Đang xử lý ({idx+1}/{len(urls)}): {url}")
        
        new_entries = process_article_to_json(url, doc_id)
        
        # Ở đây bạn có thể chèn thêm bước gọi Gemini để tạo Claim từ new_entries
        # Ví dụ: new_entries = generate_claims_with_gemini(new_entries)
        
        all_data.extend(new_entries)

    # 4. Lưu lại toàn bộ vào file JSON (Ghi đè để đảm bảo đúng định dạng list)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Hoàn thành! Đã cập nhật {output_file}. Tổng cộng có {len(all_data)} đoạn văn.")

if __name__ == "__main__":
    main()