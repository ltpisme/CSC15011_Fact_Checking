import json
import random
import re
import requests
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CẤU HÌNH ---
OPENROUTER_API_KEY = ""
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Cấu hình Model
MODELS = {
    "generator": "google/gemini-2.5-flash-lite",
    "voters": [
        "openai/gpt-4o-mini",
        "qwen/qwen-2.5-72b-instruct",
        "mistralai/mistral-small-24b-instruct-2501",
    ]
}

# --- HÀM TRUY XUẤT (RETRIEVAL) ---
def retrieve_top_k(claim, kb_data, k=3):
    texts = [item['text'] for item in kb_data]
    vectorizer = TfidfVectorizer().fit(texts + [claim])
    vectors = vectorizer.transform(texts)
    claim_vector = vectorizer.transform([claim])
    cosine_similarities = cosine_similarity(claim_vector, vectors).flatten()
    related_indices = cosine_similarities.argsort()[-k:][::-1]
    
    return [{"text": kb_data[idx]['text'], "source": kb_data[idx].get('source', 'Unknown')} for idx in related_indices]

# --- HÀM GỌI AI ---

def call_openrouter(model, system_prompt, user_prompt):
    global total_session_cost
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1000 # Đủ để lấy Label + Evidence Quote
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=40)
        return response.json()['choices'][0]['message']['content'].strip()
    except: return "ERROR | ERROR"

# --- PIPELINE CHÍNH ---
def run_factcheck_pipeline(kb_path):
    with open(kb_path, 'r', encoding='utf-8') as f:
        kb_data = json.load(f)

    seed_context = random.choice(kb_data)
    print(f"🌱 Seed Context: {seed_context['id']}")

    # --- STAGE 1: Sinh Claim với Thực thể đầy đủ ---
    gen_system = "You are an NLP researcher. Output JSON ONLY: {\"claims\": [{\"claim\": \"...\", \"label\": \"...\"}]}"
    gen_user = (
        f"Dựa trên văn bản: {seed_context['text']}\n"
        "Tạo 3 claims (NEI, REFUTED, NEI). Yêu cầu: Câu hoàn chỉnh, đầy đủ thực thể (tên, ngày tháng, địa điểm). "
        "Không dùng 'họ', 'đối tượng'. Ví dụ: 'Tòa án Ôn Châu tuyên án tử hình 11 người vào ngày 29/1/2026'."
    )
    
    raw_gen = call_openrouter(MODELS['generator'], gen_system, gen_user)
    
    # In kết quả thô để xem AI có thực sự trả về gì không
    print(f"🤖 Raw Generator Output: {raw_gen}") 

    try:
        # Sử dụng findall hoặc check None để tránh lỗi .group()
        match = re.search(r'\{.*\}', raw_gen, re.DOTALL)
        if match:
            json_str = match.group()
            generated_claims = json.loads(json_str).get("claims", [])
            print(f"✅ Đã tạo {len(generated_claims)} claims.")
        else:
            print("❌ AI không trả về định dạng JSON hợp lệ.")
            return # Hoặc xử lý tiếp tùy bạn
            
    except Exception as e:
        print(f"❌ Lỗi tại Stage 1: {e}")
        return

    final_dataset = []
    vote_system = "Answer format: LABEL | EXACT_QUOTE_FROM_SOURCES. Labels: SUPPORT, REFUTED, NEI."

    for item in generated_claims[:3]:
        claim = item['claim']
        label_gen = item['label'].upper()
        print(f"\n🔍 Checking: {claim}")

        # Retrieval
        contexts = retrieve_top_k(claim, kb_data)
        context_block = "\n".join([f"S{i+1}: {c['text']}" for i, c in enumerate(contexts)])
        print(contexts)
        user_prompt = f"SOURCES:\n{context_block}\n\nCLAIM: {claim}"

        # --- VOTING: 3 Models ---
        votes_data = []

        # Gọi 3 model vote
        for model in MODELS['voters']:
            res = call_openrouter(model, vote_system, user_prompt)
            label = res.split('|')[0].strip().upper()
            quote = res.split('|')[1].strip() if '|' in res else res
            votes_data.append({"model": model, "vote": label, "quote": quote})

        # Tính kết quả vote đa số
        all_labels = [v['vote'] for v in votes_data]
        count_all = Counter(all_labels)
        final_label, final_freq = count_all.most_common(1)[0]
        
        if final_freq == 3:
            status = "HIGH_CONFIDENCE"
            qc_tag = "TYPE_A"
        elif final_freq == 2:
            status = "MAJORITY"
            qc_tag = "TYPE_B"
        else:
            status = "CONFLICT"
            qc_tag = "TYPE_C"
        
        print(f"🗳️ Voting Result: {final_freq}/3. Label: {final_label} ({all_labels})")

        final_dataset.append({
            "claim": claim,
            "final_label": final_label,
            "status": status,
            "qc_tag": qc_tag,
            "consensus_score": f"{Counter([v['vote'] for v in votes_data]).most_common(1)[0][1]}/{len(votes_data)}",
            "evidence": contexts,
            "voter_details": votes_data
        })

    # Save
    with open('tiered_factcheck_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    run_factcheck_pipeline("knowledge_base.json")
    
