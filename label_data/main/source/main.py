"""
Module: Claim generation and labeling-voting system

Description: Nơi thực hiện pipeline chính, sử dụng model
            đê sinh claim, dùng model retrival để lấy các đoạn văn bản liên quan.
            Và dùng llms voting pipeline để dán nhãn các claim.
"""

import os
import requests
from dotenv import load_dotenv
import random
import json
from configs import CLAIM_GEN_SYSTEM_PROMPT, CLAIM_GEN_USER_PROMPT_TEMPLATE, VOTING_SYSTEM_PROMPT, VOTING_USER_PROMPT, LIST_OF_MODELS



load_dotenv()
MY_API_KEY = os.getenv("API_KEY")
KB_PATH = os.getenv("KNOWLEDGE_BASE_PATH")
MODELS = LIST_OF_MODELS



print(KB_PATH)



def call_openrouter(model,system_prompt, user_prompt, temperature, max_token=1000):

    payload = {
        "model": model,
        "messages": [
            {"role":"system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_token
    }

    headers = {"Authorization": f"Bearer {MY_API_KEY}", "Content-Type": "application/json"}

    try:
        response = requests.post(payload, headers=headers, json=payload, timeout=40)
    except:
        return "ERROR"
    

def run_factcheck_pipeline():
    with open(KB_PATH, "r", encoding='utf-8') as f:
        kb_data = json.load(f)
    
    #1. Tạo claim
    
    seed_context = random.choice(kb_data)

    gen_system = CLAIM_GEN_SYSTEM_PROMPT
    gen_user = CLAIM_GEN_USER_PROMPT_TEMPLATE.format(context={seed_context})
    
    raw_gen = call_openrouter(MODELS['generator'], gen_system, gen_user, 0.7)

    #2. Retrival các đoạn văn bản liên quan


    #3. Dán nhãn sử dụng LLMs voting system

