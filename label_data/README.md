# label_data — Pipeline Tạo & Dán Nhãn Dữ Liệu Fact-checking

Thư mục này chứa toàn bộ pipeline để **tự động tạo và dán nhãn dữ liệu** cho bài toán kiểm chứng thông tin (Fact-checking) tiếng Việt. Pipeline được xây dựng theo kiến trúc RAG (Retrieval-Augmented Generation) kết hợp cơ chế **LLM Voting** để đảm bảo chất lượng nhãn.

---

## Cấu trúc thư mục

```
label_data/
├── demo/                        # Phiên bản thử nghiệm (prototype)
│   ├── create_kb.py             # Crawl bài báo & xây dựng Knowledge Base
│   ├── generate_data.py         # Pipeline tạo + dán nhãn claim (TF-IDF retrieval)
│   ├── knowledge_base.json      # Knowledge Base đầu ra
│   └── tiered_factcheck_dataset.json  # Dataset đã gán nhãn
│
└── main/                        # Phiên bản chính thức (production)
    ├── pyproject.toml           # Cấu hình dự án & dependencies
    ├── data/
    │   └── kb_embeddings.npy    # Cache embedding BGE-M3 (pre-computed)
    └── source/
        ├── configs.py           # Quản lý tập trung prompts & cấu hình model
        ├── build_cache_bge-m3.py  # Xây dựng cache embedding BGE-M3
        ├── content_retrieval.py   # Module truy xuất bằng chứng (BGE-M3 & TF-IDF)
        └── main.py              # Pipeline chính
```

---

## Tổng quan Pipeline

Pipeline hoạt động theo **3 giai đoạn chính**:

```
[Bài báo Online]
      │
      ▼
┌─────────────────┐
│  1. Xây dựng KB │  Crawl → Chunking → knowledge_base.json
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. Sinh Claim  │  Seed Context → LLM Generator → {SUPPORTED, REFUTED, NEI}
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Dán nhãn    │  Retrieval (BGE-M3/TF-IDF) → LLM Voting (3 models)
└─────────────────┘
```

### Giai đoạn 1 — Xây dựng Knowledge Base

- **Script:** `demo/create_kb.py`
- Đọc danh sách URL từ file `newspaper_url.txt`
- Sử dụng **Trafilatura** để crawl và trích xuất nội dung bài báo
- Chia nhỏ văn bản thành các chunk (~1500 ký tự) bằng `RecursiveCharacterTextSplitter`
- Lưu kết quả vào `knowledge_base.json` (hỗ trợ ghi nối tiếp)

### Giai đoạn 2 — Sinh Claim

- **Script:** `main/source/main.py`
- Chọn ngẫu nhiên một đoạn văn (**Seed Context**) từ Knowledge Base
- Gọi LLM Generator (`google/gemini-2.5-flash-lite`) để tạo **3 claims** với 3 nhãn:
  - `SUPPORTED` — Thông tin khớp hoàn toàn với văn bản gốc
  - `REFUTED` — Thông tin bị sai lệch (thay đổi số liệu, đảo ngược hành động…)
  - `NEI` — Thông tin liên quan đến thực thể nhưng không đủ bằng chứng kiểm chứng

### Giai đoạn 3 — Truy xuất & Dán nhãn (LLM Voting)

- **Module Retrieval** (`main/source/content_retrieval.py`) hỗ trợ 2 phương pháp:

  | Phương pháp | Mô hình | Ưu điểm |
  |-------------|---------|---------|
  | **Dense** | `BAAI/bge-m3` | Tìm kiếm ngữ nghĩa đa ngôn ngữ |
  | **Sparse** | TF-IDF (sklearn) | Chính xác với thực thể, tên riêng, số liệu |

- **Voting Phase:** 3 LLM voter chấm nhãn độc lập dựa trên bằng chứng được truy xuất, nhãn cuối cùng được quyết định bằng **majority voting**:

  | Vai trò | Model |
  |---------|-------|
  | Voter 1 | `openai/gpt-4o-mini` |
  | Voter 2 | `qwen/qwen-2.5-72b-instruct` |
  | Voter 3 | `mistralai/mistral-small-24b-instruct-2501` |

---

## Cài đặt (thư mục `main/`)

Dự án sử dụng **uv** để quản lý môi trường ảo.

```bash
# Tạo môi trường ảo và cài dependencies
cd label_data/main
uv sync
```

**Các thư viện chính:**

| Thư viện | Mục đích |
|---------|---------|
| `flagembedding` | Mô hình BGE-M3 để tính embedding |
| `numpy` | Tính toán cosine similarity trên ma trận embedding |
| `python-dotenv` | Nạp biến môi trường (API key, đường dẫn KB) |
| `trafilatura` | Crawl & trích xuất nội dung bài báo |
| `langchain-text-splitters` | Chia văn bản thành chunk |
| `scikit-learn` | TF-IDF vectorizer |

---

## Cấu hình

Tạo file `.env` trong thư mục `main/` với nội dung:

```env
API_KEY=your_openrouter_api_key
KNOWLEDGE_BASE_PATH=../path/to/knowledge_base.json
```

> API key lấy từ [OpenRouter](https://openrouter.ai/) — nền tảng tổng hợp nhiều LLM qua một endpoint duy nhất.

---

## Sử dụng



### 1. Build cache embedding BGE-M3 (chạy một lần)

```bash
cd main
python source/build_cache_bge-m3.py
```

Cache được lưu tại `data/kb_embeddings.npy`, tái sử dụng cho các lần chạy tiếp theo.

### 2. Chạy pipeline chính

```bash
cd main
python source/main.py
```

---

## Định dạng dữ liệu đầu ra (chưa chốt)

File `tiered_factcheck_dataset.json` chứa danh sách các mẫu có cấu trúc:

```json
[
  {
    "claim": "Tòa án Ôn Châu tuyên án tử hình 11 người vào ngày 29/1/2026.",
    "label": "SUPPORTED",
    "evidence": [
      {
        "text": "...(đoạn bằng chứng liên quan)...",
        "source": "https://..."
      }
    ],
    "voting_results": {
      "gpt-4o-mini": "SUPPORTED",
      "qwen-2.5-72b": "SUPPORTED",
      "mistral-small": "SUPPORTED"
    }
  }
]
```

---

## So sánh Demo vs Main

| Tiêu chí | `demo/` | `main/` |
|----------|---------|---------|
| Retrieval | TF-IDF | BGE-M3 + TF-IDF (so sánh) |
| Cấu hình | Hardcode trong file | Tập trung tại `configs.py` + `.env` |
| Prompts | Inline trong code | Module riêng (`configs.py`) |
| Mục đích | Thử nghiệm nhanh | Production-ready |
