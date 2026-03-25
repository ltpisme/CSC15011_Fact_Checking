# Tổng quan Pipeline — Fact-checking Label Generation

Tài liệu này tổng hợp toàn bộ **model**, **thư viện**, và **luồng xử lý** của pipeline
`label_data/main/` để tạo và dán nhãn dữ liệu Fact-checking tự động.

---

## 1. Kiến trúc tổng thể

```
Knowledge Base (.jsonl)
        │
        ▼
[build_cache_bge-m3.py]   ─── chỉ chạy 1 lần ───►  kb_embeddings.npy
        │
        ▼
[main.py] ─── run_factcheck_pipeline()
   │
   ├─ Bước 1: Sinh Claim
   │       └─ LLM Generator (Gemini 2.5 Flash Lite)
   │           ─► 3 claims: SUPPORTED / REFUTED / NEI
   │
   ├─ Bước 2: Truy xuất Bằng chứng (Evidence Retrieval)
   │       └─ Hybrid BM25 + BGE-M3
   │           ─► top-3 đoạn văn liên quan nhất từ KB
   │
   └─ Bước 3: LLM Voting (Dán nhãn)
           └─ 3 voter models biểu quyết độc lập
               ─► Majority vote ─► final_label + QC metadata
```

---

## 2. Các Model sử dụng

### 2.1 Model Sinh Claim (Generator)

| Vai trò | Model | Provider | Ghi chú |
|---|---|---|---|
| Claim Generator | `google/gemini-2.5-flash-lite` | Google (qua OpenRouter) | `temperature=0.7`, sinh 3 claims/context |

> Được gọi qua **OpenRouter API**: `https://openrouter.ai/api/v1/chat/completions`

### 2.2 Model Embedding (Retrieval)

| Vai trò | Model | Thư viện | Ghi chú |
|---|---|---|---|
| Dense Embedding | `BAAI/bge-m3` | `FlagEmbedding` | Đa ngôn ngữ, `use_fp16=True`, lazy-load singleton |

- Vector KB được **pre-compute** 1 lần và lưu thành `kb_embeddings.npy`.
- Tại inference, chỉ encode claim rồi dot-product với ma trận cache → cực nhanh.

### 2.3 Models Voting (Labeller)

Ba model biểu quyết **độc lập** với nhau (`temperature=0.0`):

| # | Model | Provider |
|---|---|---|
| 1 | `openai/gpt-4o-mini` | OpenAI |
| 2 | `qwen/qwen-2.5-72b-instruct` | Alibaba Qwen |
| 3 | `mistralai/mistral-small-24b-instruct-2501` | Mistral AI |

> Nhãn cuối cùng = **majority vote** (đa số). Gắn kèm QC metadata:
> - `TYPE_A` / `HIGH_CONFIDENCE` — 3/3 đồng thuận
> - `TYPE_B` / `MEDIUM_CONFIDENCE` — 2/3 đồng thuận
> - `TYPE_C` / `LOW_CONFIDENCE` — không có đa số (0/3 hoặc 1/3)

---

## 3. Các Thư viện chính

### 3.1 Dependencies (khai báo trong `pyproject.toml`)

| Thư viện | Phiên bản | Mục đích sử dụng |
|---|---|---|
| `flagembedding` | `>=1.3.5` | Load & chạy model `BAAI/bge-m3` |
| `torch` | `>=2.10.0` | Backend cho FlagEmbedding / Transformers |
| `transformers` | `>=4.40.0, <5.0.0` | Dependency của FlagEmbedding |
| `numpy` | `>=2.4.3` | Lưu/load `.npy`, dot-product vector |
| `scikit-learn` | `>=1.8.0` | `TfidfVectorizer`, `cosine_similarity` |
| `requests` | `>=2.32.5` | Gọi HTTP tới OpenRouter API |
| `python-dotenv` | `>=1.2.2` | Đọc API key từ file `.env` |

### 3.2 Dependencies dùng trong code (cần cài thêm)

| Thư viện | Mục đích sử dụng | Lệnh cài |
|---|---|---|
| `rank_bm25` | BM25 sparse retrieval (bước 1 của Hybrid) | `uv add rank-bm25` |
| `pyvi` | Vietnamese tokenizer (`ViTokenizer`) dùng cho BM25 | `uv add pyvi` |
| `tqdm` | Progress bar trong `run_factcheck_pipeline()` | `uv add tqdm` |

> **Lưu ý:** Ba thư viện trên chưa được khai báo trong `pyproject.toml`.
> Chạy `uv sync` sau khi thêm để đồng bộ môi trường.

---

## 4. Chiến lược Retrieval

Ba phương pháp được cài đặt trong `content_retrieval.py`, có thể hoán đổi cho nhau:

| Phương pháp | Hàm | Ưu điểm | Nhược điểm |
|---|---|---|---|
| **TF-IDF** | `retrieve_tfidf()` | Không cần model/cache, nhanh, chính xác với số liệu/tên riêng | Không hiểu ngữ nghĩa sâu |
| **BGE-M3 Dense** | `retrieve_bge_m3()` | Hiểu ngữ nghĩa, đa ngôn ngữ | Dễ bỏ sót khi thực thể bị paraphrase |
| **Hybrid BM25 + BGE-M3** | `retrieve_hybrid_bge()` | Cân bằng: BM25 lọc n ứng viên, BGE-M3 rerank k tốt nhất | Phức tạp hơn, cần cả 2 bước |

**Pipeline chính thức dùng: `retrieve_hybrid_bge()`** với `n=30, k=3`.

---

## 5. Cấu hình môi trường

File `.env` tại thư mục **gốc của repo**:

```env
API_KEY=your_openrouter_api_key

KNOWLEDGE_BASE_PATH=label_data/main/data/KB.jsonl
KNOWLEDGE_EMBED_BASE_PATH=label_data/main/data/kb_embeddings.npy
OUTPUT_PATH=label_data/main/data/output_dataset.json
FAILED_LOG_PATH=label_data/main/data/failed_log.json
CLAIM_CONTEXT_PATH=label_data/main/data/context_claim_gen.jsonl
```

---

## 6. Thứ tự chạy Pipeline

```bash
# 1. Cài dependencies
cd label_data
uv sync

# 2. Build cache embedding (chỉ cần chạy 1 lần)
uv run python main/source/build_cache_bge-m3.py

# 3. Chạy pipeline chính
uv run python main/source/main.py
```

---

## 7. Sơ đồ phụ thuộc module

```
main.py
 ├── configs.py          ← prompts, paths, LIST_OF_MODELS
 └── content_retrieval.py
      ├── FlagEmbedding   (BAAI/bge-m3)
      ├── sklearn         (TF-IDF)
      ├── rank_bm25       (BM25)
      └── pyvi            (ViTokenizer)
```
