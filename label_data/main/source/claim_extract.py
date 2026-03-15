"""
Module: Claim Extraction
Description: Sinh n claim từ KB_vnexpress.jsonl bằng hàm generate_claims,
             xuất ra file JSON để phục vụ đánh giá retrieval hoặc labeling.
"""

import json
import random
from pathlib import Path

from configs import KB_PATH, BASE_DIR
from main import generate_claims

KB_VNEXPRESS_PATH = BASE_DIR / "label_data/main/data/context_claim_gen.jsonl"
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "claims_output.json"


def extract_claims(n_contexts: int = 10, output_path: str | Path = DEFAULT_OUTPUT_PATH) -> list[dict]:
    """Sinh claims từ n_contexts context ngẫu nhiên lấy từ KB_vnexpress.jsonl.

    Args:
        n_contexts:  Số context seed dùng để sinh claim.
        output_path: Đường dẫn file JSON đầu ra.

    Returns:
        Danh sách tất cả claim đã sinh.
    """
    with open(KB_VNEXPRESS_PATH, "r", encoding="utf-8") as f:
        vnexpress_data = [json.loads(line) for line in f if line.strip()]

    seed_contexts = random.sample(vnexpress_data, min(n_contexts, len(vnexpress_data)))

    all_claims = []
    claim_count = 0

    for ctx_idx, item in enumerate(seed_contexts):
        seed_text = item["text"]
        print(f"[Context {ctx_idx + 1}/{n_contexts}] Sinh claim từ: {seed_text[:80]}...")

        claims = generate_claims(seed_text)
        if not claims:
            print("  -> Không sinh được claim, bỏ qua.")
            continue

        for claim_item in claims:
            claim_text = claim_item.get("claim", "")
            if not claim_text:
                continue
            claim_count += 1
            all_claims.append({
                "claim_id": f"ctx{ctx_idx + 1}_claim{claim_count}",
                "claim": claim_text,
                "label": claim_item.get("label", ""),
                "seed_context_id": item.get("id", f"ctx_{ctx_idx + 1}"),
                "seed_context": seed_text,
            })
            print(f"  [{claim_count}] [{claim_item.get('label', '?')}] {claim_text[:80]}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_claims, f, ensure_ascii=False, indent=2)

    print(f"\nĐã sinh {claim_count} claims từ {n_contexts} contexts.")
    print(f"Đã xuất: {output_path}")
    return all_claims


if __name__ == "__main__":
    extract_claims(n_contexts=127, output_path=DEFAULT_OUTPUT_PATH)
