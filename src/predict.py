import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os

# Force single thread for speed on CPU
torch.set_num_threads(1) 
os.environ["OMP_NUM_THREADS"] = "1"

def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0: continue
        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue
        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end
    if current_label is not None:
        spans.append((current_start, current_end, current_label))
    return spans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--device", default="cpu") # Force CPU for quantization
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)

    # --- OPTIMIZATION: QUANTIZATION ---
    if args.device == "cpu":
        print("Quantizing model for CPU inference...")
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    # ----------------------------------

    model.to(args.device)
    model.eval()

    results = {}
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]
            enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=args.max_length, return_tensors="pt")
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                pred_ids = out.logits[0].argmax(dim=-1).cpu().tolist()
            
            spans = bio_to_spans(text, offsets, pred_ids)
            ents = [{"start": s, "end": e, "label": l, "pii": label_is_pii(l)} for s, e, l in spans]
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()