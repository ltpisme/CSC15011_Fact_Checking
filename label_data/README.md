# label_data — Pipeline Tạo & Dán Nhãn Dữ Liệu Fact-checking

## Cấu trúc thư mục

```
label_data/
├── pyproject.toml               # Cấu hình dự án & dependencies (uv)
├── .python-version              # Phiên bản Python
├── demo/                        # Phiên bản prototype
│   ├── create_kb.py             # Crawl bài báo & xây dựng Knowledge Base
│   ├── generate_data.py         # Pipeline tạo + dán nhãn claim (dùng TF-IDF)
│   ├── knowledge_base.json      # Knowledge Base đầu ra
│   └── tiered_factcheck_dataset.json  # Dataset mẫu đã gán nhãn
│
└── main/                        # Pipeline chính thức
    ├── data/
    │   ├── knowledge_base.json  # Knowledge Base
    │   ├── kb_embeddings.npy    # Cache vector BGE-M3 (pre-computed)
    │   └── output_dataset.json  # Dataset output (được ghi dần sau mỗi claim)
    └── source/
        ├── configs.py           # Prompts, đường dẫn, danh sách model
        ├── build_cache_bge-m3.py  # Build cache embedding cho Knowledge Base
        ├── content_retrieval.py   # Truy xuất bằng chứng (BGE-M3 + TF-IDF)
        └── main.py              # Pipeline chính
```

---

## Chức năng các file Python

### `demo/create_kb.py`
Crawl danh sách URL từ file `newspaper_url.txt`, trích xuất nội dung bằng **Trafilatura**, chia thành các chunk (~1500 ký tự) bằng `RecursiveCharacterTextSplitter`, lưu vào `knowledge_base.json`.

### `demo/generate_data.py`
Phiên bản prototype của pipeline chính. Dùng **TF-IDF** để truy xuất bằng chứng, không cần cache embedding. Thích hợp để thử nghiệm nhanh.

### `main/source/configs.py`
Quản lý tập trung toàn bộ cấu hình: system prompts, user prompt templates cho giai đoạn sinh claim và voting, danh sách model, và các đường dẫn file (KB, embedding cache, output).

### `main/source/build_cache_bge-m3.py`
Load `knowledge_base.json`, encode toàn bộ text bằng model `BAAI/bge-m3`, lưu ma trận vector ra `kb_embeddings.npy`. Chỉ cần chạy một lần khi KB thay đổi.

### `main/source/content_retrieval.py`
Truy xuất top-K đoạn bằng chứng liên quan tới một claim. Hỗ trợ 3 chiến lược:
- **`retrieve_top_k()`** — Sparse retrieval dùng TF-IDF (sklearn). Không cần model hay cache, phù hợp kiểm thử nhanh và bắt chính xác thực thể (tên, số liệu).
- **`retrieve_bge_m3()`** — Dense retrieval dùng cosine similarity trên cache `.npy`; model `BAAI/bge-m3` được lazy-load dưới dạng singleton.
- **`retrieve_hybrid_bge()`** — Two-stage retrieval: BM25 (ViTokenizer) lọc n ứng viên trước, BGE-M3 rerank lấy k kết quả cuối. Cân bằng giữa chính xác thực thể và hiểu ngữ nghĩa.
- **`compare_retrieval()`** — In bảng so sánh song song kết quả BGE-M3 vs TF-IDF để đánh giá chiến lược retrieval.

### `main/source/main.py`
Pipeline chính, chạy end-to-end:
1. **Sinh claim** — Chọn ngẫu nhiên seed context từ KB, gọi LLM generator tạo 3 claims (`SUPPORT` / `REFUTED` / `NEI`)
2. **Truy xuất bằng chứng** — Dùng BGE-M3 lấy top-3 đoạn liên quan
3. **LLM Voting** — 3 model voter chấm nhãn độc lập, kết quả cuối theo majority vote kèm QC metadata (`HIGH/MEDIUM/LOW_CONFIDENCE`, `TYPE_A/B/C`)
4. **Lưu kết quả** — Ghi vào `output_dataset.json` ngay sau mỗi claim (atomic write qua `.tmp`). Nếu chạy lại, các claim đã xử lý sẽ được bỏ qua tự động.

---

## Cài đặt

```bash
cd label_data
uv sync
```

Tạo file `.env` tại thư mục gốc của repo:

```env
API_KEY=your_openrouter_api_key
KNOWLEDGE_BASE_PATH=label_data/main/data/knowledge_base.json
KNOWLEDGE_EMBED_BASE_PATH=label_data/main/data/kb_embeddings.npy
OUTPUT_PATH=label_data/main/data/output_dataset.json
```

---

## Cách chạy

### 1. Build cache embedding (chỉ cần chạy 1 lần)

```bash
cd label_data/main/source
uv run build_cache_bge-m3.py
```

### 2. Chạy pipeline chính

```bash
cd label_data/main/source
uv run main.py
```

Pipeline sẽ tự động nối tiếp từ điểm dừng nếu `output_dataset.json` đã tồn tại.

### 3. Chạy prototype (không cần cache)

```bash
cd label_data/demo
python generate_data.py
```
