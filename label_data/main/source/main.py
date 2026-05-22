"""
Module: Claim generation and labeling-voting system

Description: Nơi thực hiện pipeline chính, sử dụng model
            để sinh claim, dùng model retrieval để lấy các đoạn văn bản liên quan.
            Và dùng llms voting pipeline để dán nhãn các claim.
"""

import os
import json
import logging
import random
import re
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import requests
from dotenv import load_dotenv

USE_AUDIT_RETRIEVAL = false

from configs import (
    API_URL,
    CLAIM_GEN_SYSTEM_PROMPT,
    CLAIM_GEN_USER_PROMPT_TEMPLATE,
    VOTING_SYSTEM_PROMPT,
    VOTING_USER_PROMPT,
    LIST_OF_MODELS,
    KB_PATH,
    KB_EMBED_PATH,
    OUTPUT_PATH,
    FAILED_LOG_PATH,
    CLAIM_PATH
)
from content_retrieval import retrieve_bge_m3, retrieve_hybrid_bge, retrieve_tfidf

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MY_API_KEY = os.getenv("API_KEY")

print(KB_PATH)
print(KB_EMBED_PATH)
print(CLAIM_PATH)
def call_openrouter(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int = 1000,
) -> dict | str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {MY_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=40)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error("API call failed for model %s: %s", model, e)
        return "ERROR"


def parse_json_from_response(api_response: dict | str) -> dict | None:
    """Extract and parse JSON content from an OpenRouter API response."""
    if api_response == "ERROR":
        return None
    try:
        raw_content = api_response["choices"][0]["message"]["content"]
        json_match = re.search(r"```json\s*(.*?)\s*```", raw_content, re.DOTALL)
        json_str = json_match.group(1) if json_match else raw_content
        return json.loads(json_str)
    except (KeyError, json.JSONDecodeError) as e:
        logger.error("Failed to parse JSON from response: %s", e)
        return None


def generate_claims(seed_text: str) -> list[dict]:
    """Generate SUPPORTED/REFUTED/NEI claims from a seed context."""
    gen_user = CLAIM_GEN_USER_PROMPT_TEMPLATE.format(context=seed_text)
    raw_response = call_openrouter(
        LIST_OF_MODELS["generator"], CLAIM_GEN_SYSTEM_PROMPT, gen_user, temperature=0.7
    )
    data = parse_json_from_response(raw_response)
    if data is None:
        return []
    return data.get("claims", [])


def _compute_qc_metadata(votes: list[str], num_voters: int) -> tuple[str, str, str]:
    """Return (final_label, status, qc_tag, consensus_score) from a vote list."""
    if not votes:
        return "NEI", "LOW_CONFIDENCE", "TYPE_C", f"0/{num_voters}"

    final_label, count = Counter(votes).most_common(1)[0]
    consensus_score = f"{count}/{num_voters}"

    if count == num_voters:
        return final_label, "HIGH_CONFIDENCE", "TYPE_A", consensus_score
    if count >= num_voters / 2:
        return final_label, "MEDIUM_CONFIDENCE", "TYPE_B", consensus_score
    return final_label, "LOW_CONFIDENCE", "TYPE_C", consensus_score


def vote_on_claim(claim: str, evidences: list[dict]) -> dict:
    """Run multi-model voting on a single claim and return aggregated result."""
    evidence_texts = [e["text"] for e in evidences[:3]]
    while len(evidence_texts) < 3:
        evidence_texts.append("")

    user_prompt = VOTING_USER_PROMPT.format(
        claim=claim,
        evidence_1=evidence_texts[0],
        evidence_2=evidence_texts[1],
        evidence_3=evidence_texts[2],
    )

    voter_details = []
    for voter_model in LIST_OF_MODELS["voters"]:
        response = call_openrouter(
            voter_model, VOTING_SYSTEM_PROMPT, user_prompt, temperature=0.0
        )
        parsed = parse_json_from_response(response)
        if parsed and "nhan" in parsed:
            voter_details.append({
                "model": voter_model,
                "vote": parsed["nhan"],
                "quote": parsed.get("trich_dan", ""),
            })

    votes = [v["vote"] for v in voter_details]
    final_label, status, qc_tag, consensus_score = _compute_qc_metadata(
        votes, len(LIST_OF_MODELS["voters"])
    )
    return {
        "final_label": final_label,
        "status": status,
        "qc_tag": qc_tag,
        "consensus_score": consensus_score,
        "voter_details": voter_details,
    }


def _load_existing_results() -> tuple[list[dict], set[str]]:
    """Load previously saved results and return (results, set_of_processed_claims)."""
    if not OUTPUT_PATH.exists():
        return [], set()
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        processed = {item["claim"] for item in existing}
        logger.info("Resuming — %d claims already processed.", len(existing))
        return existing, processed
    except (json.JSONDecodeError, KeyError):
        logger.warning("Output file is corrupt or unreadable, starting fresh.")
        return [], set()


def _save_all_results(results: list[dict]) -> None:
    """Ghi toàn bộ list results vào file một cách an toàn (Atomic Write)."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = OUTPUT_PATH.with_suffix(".tmp")
    
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    tmp_path.replace(OUTPUT_PATH)


def _load_failed_log() -> list[dict]:
    """Load danh sách các claim đã thất bại từ lần chạy trước."""
    if not FAILED_LOG_PATH.exists():
        return []
    try:
        with open(FAILED_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        logger.warning("Failed log file is corrupt, starting fresh.")
        return []


def _save_failed_log(failed: list[dict]) -> None:
    """Ghi danh sách claim thất bại ra file một cách an toàn (Atomic Write)."""
    if not failed:
        return
    FAILED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = FAILED_LOG_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=4)
    tmp_path.replace(FAILED_LOG_PATH)


from tqdm import tqdm
from datetime import datetime, timezone

def run_factcheck_pipeline(n=127, batch_size=20) -> list[dict]:
    """Load Knowledge Base"""
    kb_data = []
    with open(KB_PATH, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    kb_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    kb_embeddings = np.load(KB_EMBED_PATH)
    print("--- HOÀN THÀNH LẤY KB ---")

    """Load Existing Results & Filter Used Contexts"""
    # 1. Load các claim đã làm
    results, processed_claims = _load_existing_results()
    failed_logs = _load_failed_log()
    
    # 2. Xây dựng tập hợp các context ĐÃ DÙNG để lọc (Tránh gọi lại LLM vô ích)
   
    used_context_texts = {r.get("seed_context", "") for r in results if "seed_context" in r}

    """Load and Filter Contexts"""
    all_contexts = []
    with open(CLAIM_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ctx = json.loads(line)
            # Lọc ngay từ lúc đọc file: Chỉ giữ context chưa làm
            if ctx["text"] not in used_context_texts:
                all_contexts.append(ctx)

    # 3. Trộn ngẫu nhiên các context CÒN LẠI (đảm bảo không trùng & ngẫu nhiên)
    random.shuffle(all_contexts)
    
    # Tính số lượng thực tế cần chạy
    actual_n = min(n, len(all_contexts))
    
    print(f"--- Tổng context còn lại chưa xử lý: {len(all_contexts)} ---")
    if actual_n == 0:
        print("Không còn context mới nào để xử lý. Dừng pipeline.")
        return results

    new_claims_count = 0

    # Khởi tạo progress bar
    pbar = tqdm(total=actual_n, desc="Fact-checking Pipeline", unit="ctx")

    for i in range(actual_n):
        seed_context = all_contexts[i]
        
        # Cập nhật thanh tiến trình
        pbar.set_postfix({"ctx": i+1, "new_claims": new_claims_count})

        # --- BƯỚC GEN CLAIM ---
        claims = generate_claims(seed_context["text"])
        if not claims:
            failed_logs.append({
                "claim": None,
                "seed_context": seed_context["text"][:200],
                "stage": "claim_generation",
                "error": "LLM trả về danh sách claims rỗng",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            pbar.update(1)
            continue

        # --- XỬ LÝ TỪNG CLAIM ---
        for claim_item in claims:
            claim_text = claim_item["claim"]

            # Double-check để chống trùng Claim (đề phòng 2 context gen ra claim giống nhau)
            if claim_text in processed_claims:
                continue

            try:
                # Bước 1: Truy xuất bằng chứng
                evidences = retrieve_hybrid_bge(claim_text, kb_data, kb_embeddings)

                # Bước 2: Voting
                voting_result = vote_on_claim(claim_text, evidences)

                if voting_result["consensus_score"].startswith("0/"):
                    failed_logs.append({
                        "claim": claim_text,
                        "stage": "voting",
                        "error": "Tất cả voters trả về lỗi",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    continue

                # Đóng gói kết quả (Quan trọng: Lưu lại seed_context để lần sau lọc)
                result = {
                    "claim": claim_text,
                    "seed_context": seed_context["text"],  
                    "final_label": voting_result["final_label"],
                    "status": voting_result["status"],
                    "qc_tag": voting_result["qc_tag"],
                    "consensus_score": voting_result["consensus_score"],
                    "evidence": [{"text": e["text"], "source": e["source"]} for e in evidences],
                    "voter_details": voting_result["voter_details"],
                }

                results.append(result)
                processed_claims.add(claim_text)
                new_claims_count += 1

                # KIỂM TRA BATCH: Lưu file sau mỗi 'batch_size' claim MỚI
                if new_claims_count % batch_size == 0:
                    _save_all_results(results)
                    _save_failed_log(failed_logs)

            except Exception as e:
                # logger.error(f"Lỗi khi xử lý claim: {e}") # Có thể comment lại để tránh spam log
                failed_logs.append({
                    "claim": claim_text,
                    "stage": "processing",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        # Cập nhật tiến trình sau khi xong 1 context
        pbar.update(1)

    pbar.close()

    # Lưu vét lần cuối (tránh sót các claim dư ra ngoài batch_size)
    if new_claims_count > 0:
        _save_all_results(results)
        _save_failed_log(failed_logs)

    print(f"--- Pipeline hoàn tất! Sinh mới: {new_claims_count} claims. Tổng dataset: {len(results)} ---")
    return results


if __name__ == "__main__":
    run_factcheck_pipeline(37, 5)


#2473
#15/03 - đợt 1: 379 data - score: 0.585 - price: 0.69 - tổng: 379
#15/03 - đợt 2: 750 data - score: 0.584 - price: 0.99 - tổng: 1129
#16/03 - đợt 3: 537 data - score: 0.586 - price: 1.295   - tổng: 1666
#16/03 - đợt 4:
#16/03 - đợt 5:
#17/03 - đợt 6:
#17/03 - đợt 7:
#17/03 - đợt 8: