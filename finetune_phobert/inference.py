import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# --- COPY LẠI CÁC CLASS MODEL TỪ FILE TRAIN CỦA BẠN ---
# (Đảm bảo có PhoBERTFactChecker, AttentionAggregator, CrossContextEncoder)

class FactCheckingDataset(Dataset):
    LABEL_MAP = {"SUPPORTED": 0, "NEI": 1, "REFUTED": 2}

    def __init__(self, data, model_name="vinai/phobert-base", max_length=256):
        self.data = data
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.data)

    def _get_context(self, evidence_list, idx):
        if idx < len(evidence_list):
            return evidence_list[idx].get("text", "")
        return ""

    def __getitem__(self, idx):
        item = self.data[idx]
        claim = item["claim"]
        evidences = item.get("evidence", [])

        ctx_1 = self._get_context(evidences, 0)
        ctx_2 = self._get_context(evidences, 1)
        ctx_3 = self._get_context(evidences, 2)

        label_str = item.get("final_label", item.get("label", "NEI"))
        if isinstance(label_str, str):
            label = self.LABEL_MAP.get(label_str.upper(), 1)
        else:
            label = label_str

        enc_1 = self.tokenizer(claim, ctx_1, padding="max_length", truncation=True,
                            max_length=self.max_length, return_tensors='pt')
        enc_2 = self.tokenizer(claim, ctx_2, padding="max_length", truncation=True,
                            max_length=self.max_length, return_tensors='pt')
        enc_3 = self.tokenizer(claim, ctx_3, padding="max_length", truncation=True,
                            max_length=self.max_length, return_tensors='pt')

        return {
            "input_ids_1": enc_1["input_ids"].squeeze(0),
            "attention_mask_1": enc_1["attention_mask"].squeeze(0),
            "input_ids_2": enc_2["input_ids"].squeeze(0),
            "attention_mask_2": enc_2["attention_mask"].squeeze(0),
            "input_ids_3": enc_3["input_ids"].squeeze(0),
            "attention_mask_3": enc_3["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }



class PhoBERTFactChecker(nn.Module):
    def __init__(self, model_name="vinai/phobert-base-v2", num_classes=3, dropout_prob=0.3):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        hidden = self.phobert.config.hidden_size
        
        # Thêm Attention Aggregator
        self.aggregator = AttentionAggregator(hidden_size=hidden)
    
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob / 2),
            nn.Linear(hidden // 2, num_classes)
        )

    def _encode_pair(self, input_ids, attention_mask):
        output = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        # Sử dụng token [CLS] tại index 0
        return output.last_hidden_state[:, 0, :]
    
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, input_ids_3, attention_mask_3):
        # 1. Trích xuất [CLS] token của 3 context
        cls_1 = self._encode_pair(input_ids_1, attention_mask_1)
        cls_2 = self._encode_pair(input_ids_2, attention_mask_2)
        cls_3 = self._encode_pair(input_ids_3, attention_mask_3)
    
        # 2. Gom 3 vector lại thành tensor (batch, 3, hidden)
        stacked_cls = torch.stack([cls_1, cls_2, cls_3], dim=1)
        
        # 3. Dùng Attention để tự động tính tổng có trọng số
        fused_cls, attn_weights = self.aggregator(stacked_cls) # fused_cls: (batch, hidden)
    
        # 4. Đưa qua Classifier
        logits = self.classifier(fused_cls)
    
        return logits, attn_weights

class AttentionAggregator(nn.Module):
    """
    Attention-weighted sum: model tự học context nào quan trọng nhất.
    Thay thế torch.max (quá thô, mất thông tin).
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )

    def forward(self, stacked_cls):
        # stacked_cls: (batch, num_contexts, hidden)
        attn_scores = self.attention(stacked_cls)         # (batch, 3, 1)
        attn_weights = torch.softmax(attn_scores / 0.5, dim=1) # Temperature = 0.5 làm cho trọng số dồn về 1 context mạnh hơn
        output = (stacked_cls * attn_weights).sum(dim=1)   # (batch, hidden)
        return output, attn_weights.squeeze(-1)

class CrossContextEncoder(nn.Module):
    """
    Cho 3 CLS vectors tương tác qua Self-Attention.
    Phát hiện mâu thuẫn (REFUTED) hoặc nhất quán (SUPPORTED) giữa các context.
    Giảm xuống 1 layer (thay vì 2) để phù hợp với data size nhỏ.
    """
    def __init__(self, hidden_size, num_heads=8, num_layers=1, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, stacked_cls):
        encoded = self.transformer(stacked_cls)
        return self.layer_norm(encoded + stacked_cls)  # residual


class FactCheckerInference:
    def __init__(self, model_path, model_name="vinai/phobert-large", device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 1. Khởi tạo cấu trúc model
        self.model = PhoBERTFactChecker(model_name=model_name, num_classes=3)
        
        # 2. Load trọng số từ file .pt
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.label_map = {0: "SUPPORTED", 1: "NEI", 2: "REFUTED"}
        print(f"✅ Model loaded from {model_path} on {self.device}")

    def predict(self, claim, evidences, max_length=256):
        """
        evidences: List chứa tối đa 3 chuỗi văn bản dẫn chứng
        """
        # Đảm bảo luôn có 3 context (điền rỗng nếu thiếu)
        while len(evidences) < 3:
            evidences.append("")
        
        # Tokenize từng cặp Claim + Evidence
        inputs = []
        for i in range(3):
            enc = self.tokenizer(claim, evidences[i], 
                                 padding="max_length", 
                                 truncation=True,
                                 max_length=max_length, 
                                 return_tensors='pt')
            inputs.append(enc)

        # Chuyển dữ liệu lên GPU/CPU
        input_ids = [enc['input_ids'].to(self.device) for enc in inputs]
        masks = [enc['attention_mask'].to(self.device) for enc in inputs]

        with torch.no_grad():
            # Forward qua model
            logits, attn_weights = self.model(
                input_ids[0], masks[0],
                input_ids[1], masks[1],
                input_ids[2], masks[2]
            )
            
            # Lấy xác suất và nhãn
            probs = F.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_idx].item()
            
        return {
            "label": self.label_map[pred_idx],
            "confidence": confidence,
            "attention": attn_weights.cpu().numpy()[0].tolist() # Xem model chú ý vào context nào
        }
def run_test_evaluation(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Running inference on: {device}")

    # 1. Load Dữ liệu
    with open(config['test_path'], 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    test_dataset = FactCheckingDataset(test_data, config['model_name'], config['max_length'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # 2. Khởi tạo và Load Model
    model = PhoBERTFactChecker(model_name=config['model_name'], num_classes=3)
    checkpoint = torch.load(config['save_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    label_map = {0: "SUPPORTED", 1: "NEI", 2: "REFUTED"}
    results = []

    # 3. Vòng lặp dự đoán
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Inference")):
            input_ids_1 = batch['input_ids_1'].to(device)
            mask_1 = batch['attention_mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            mask_2 = batch['attention_mask_2'].to(device)
            input_ids_3 = batch['input_ids_3'].to(device)
            mask_3 = batch['attention_mask_3'].to(device)
            true_labels = batch['label'].cpu().numpy()

            logits, attn_weights = model(input_ids_1, mask_1, input_ids_2, mask_2, input_ids_3, mask_3)
            
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            attns = attn_weights.cpu().numpy()

            # Lưu thông tin chi tiết từng mẫu
            for j in range(len(preds)):
                sample_idx = i * config['batch_size'] + j
                original_item = test_data[sample_idx]
                
                results.append({
                    "claim": original_item['claim'],
                    "true_label": label_map[true_labels[j]],
                    "predicted_label": label_map[preds[j]],
                    "confidence": float(probs[j][preds[j]]),
                    "context_weights": attns[j].tolist(),
                    "is_correct": bool(preds[j] == true_labels[j])
                })

    # 4. Xuất kết quả
    output_file = "test_results_detailed.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # Tính nhanh Accuracy
    acc = sum([1 for r in results if r['is_correct']]) / len(results)
    print(f"\n✅ Đã lưu kết quả tại: {output_file}")
    print(f"📊 Accuracy trên tập Test: {acc:.4f}")

if __name__ == "__main__":
    INFERENCE_CONFIG = {
        'test_path': 'HIGH_CONFIDENCE/test.json',
        'save_path': 'best_phobert_factcheck_v4.pt',
        'model_name': 'vinai/phobert-large',
        'max_length': 256,
        'batch_size': 16  # Có thể tăng lên 32 nếu dùng RTX 3090
    }
    run_test_evaluation(INFERENCE_CONFIG)