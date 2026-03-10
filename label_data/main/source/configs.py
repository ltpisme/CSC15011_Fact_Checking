# configs.py
"""
Module: Centralized Prompt Management
Description: Quản lý tập trung các System Prompts và User Prompt Templates 
             cho quy trình xử lý dữ liệu Fact-checking.

Cấu trúc gồm 2 giai đoạn chính:
    1. Claim Generation: Chiến thuật tạo Luận điểm từ văn bản gốc (Seed Context).
    2. Labelling (Voting Phase): Quy tắc dán nhãn SUPPORTED/REFUTED/NEI dựa trên bằng chứng.
"""
import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

load_dotenv(BASE_DIR / ".env")


KB_RELATIVE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base.json")
KB_EMBED_RELATIVE_PATH = os.getenv("KNOWLEDGE_EMBED_BASE_PATH")
OUTPUT_RELATIVE_PATH = os.getenv("OUTPUT_PATH", "label_data/main/data/output_dataset.json")
KB_PATH = BASE_DIR / KB_RELATIVE_PATH
KB_EMBED_PATH = BASE_DIR / KB_EMBED_RELATIVE_PATH
OUTPUT_PATH = BASE_DIR / OUTPUT_RELATIVE_PATH






LIST_OF_MODELS = {
    "generator": "google/gemini-2.5-flash-lite",
    "voters": [
        "openai/gpt-4o-mini",
        "qwen/qwen-2.5-72b-instruct",
        "mistralai/mistral-small-24b-instruct-2501",
    ]
}

API_URL = "https://openrouter.ai/api/v1/chat/completions"


CLAIM_GEN_SYSTEM_PROMPT = """
Bạn là một chuyên gia ngôn ngữ học chuyên tạo dữ liệu cho bài toán Fact-checking.
Nhiệm vụ của bạn là trích xuất và biến đổi thông tin từ văn bản để tạo ra các Luận điểm (Claims) có độ khó cao.

YÊU CẦU VỀ HÌNH THỨC:
1. Mỗi Claim phải là một câu khẳng định hoàn chỉnh, độc lập về ngữ cảnh.
2. Tuyệt đối không dùng đại từ nhân xưng (họ, ông ấy, đối tượng đó...). 
3. Phải bao gồm đầy đủ các thực thể: Tên riêng, Chức danh, Thời gian cụ thể, Địa điểm.

YÊU CẦU VỀ LOGIC NHÃN:
- SUPPORTED: Trích xuất thông tin khớp hoàn toàn với văn bản hoặc suy luận logic trực tiếp từ văn bản.
- REFUTED: Tạo ra một câu sai lệch hoàn toàn về sự thật nhưng vẫn giữ nguyên các thực thể quan trọng (Ví dụ: Thay đổi con số, đảo ngược hành động, thay đổi ngày tháng).
- NEI: Tạo ra một câu liên quan đến các thực thể trong văn bản nhưng thông tin chính KHÔNG hề được đề cập (Ví dụ: Thêm vào một hành động mà văn bản không nói tới).

ĐỊNH DẠNG TRẢ VỀ: Duy nhất JSON format.
"""

CLAIM_GEN_USER_PROMPT_TEMPLATE = """
Dựa trên văn bản nguồn sau:
---
{context}
---

Hãy tạo ra 3 Luận điểm (Claims) theo yêu cầu sau:
1. Nhãn SUPPORTED: Một câu khẳng định đúng hoàn toàn.
2. Nhãn REFUTED: Một câu mâu thuẫn trực tiếp với văn bản.
3. Nhãn NEI: Một câu chứa các thực thể trong bài nhưng bài không có đủ thông tin để kiểm chứng.

Trả về kết quả theo định dạng JSON:
{{
  "claims": [
    {{"claim": "Nội dung câu 1", "label": "SUPPORTED"}},
    {{"claim": "Nội dung câu 2", "label": "REFUTED"}},
    {{"claim": "Nội dung câu 3", "label": "NEI"}}
  ]
}}
"""

VOTING_SYSTEM_PROMPT =  """
Bạn là một chuyên gia kiểm chứng sự thật (Face-checker) có tư duy logic sắc bén. Nhiệm vụ của bạn là xác định tính xác thực của một “Luận điểm” dựa trên các “Bằng chứng” được cung cấp.
Quy tắc dán nhãn:
SUPPORTED: Nếu các bằng chứng chứa thông tin trực tiếp hoặc đủ để suy luận ủng hộ luận điểm.
REFUTED: Nếu các bằng chứng chứa thông tin trực tiếp hoặc phủ định luận điểm.
NEI (Not Enough Information): Nếu các bằng chứng không liên quan hoặc không đủ thông tin để khẳng định hay phủ định luận điểm.

Nguyên tắc làm việc:
Chỉ dựa vào bằng chứng được cung cấp, không dùng kiến thức bên ngoài.
Suy nghĩ từng bước (Chain-of-Thought): Phân tích mối quan hệ giữa các thực thể, mốc thời gian và hành động trong luận điểm so với bằng chứng.
Luôn trả về kết quả dưới định dạng JSON.
    """

VOTING_USER_PROMPT = """
LUẬN ĐIỂM (CONTEXT):
{claim}

CÁC BẰNG CHỨNG (EVIDENCES):
[1] {evidence_1}
[2] {evidence_2}
[3] {evidence_3}

YÊU CẦU:
Thực hiện phân tích logic và đưa ra nhãn cuối cùng. Trả về DUY NHẤT một khối JSON theo cấu trúc sau:
{{
    "nhan": "SUPPORT/REFUTED/NEI",
    "trich_dan": "Trích nguyên văn câu hoặc cụm từ quan trọng nhất từ bằng chứng ủng hộ quyết định của bạn"
}}
"""

