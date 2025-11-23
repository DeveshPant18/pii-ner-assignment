import json
import time
import argparse
import statistics
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Suppress quantization warnings
warnings.filterwarnings("ignore")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--device", default="cpu") 
    args = ap.parse_args()

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)

    # --- QUANTIZATION (The Speed Hack) ---
    if args.device == "cpu":
        print("Quantizing model for CPU inference...")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    # -------------------------------------

    model.to(args.device)
    model.eval()

    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    times_ms = []
    MAX_LEN = 64  # Keep this optimization

    print("Warming up...")
    for _ in range(10):
        enc = tokenizer(texts[0], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            _ = model(input_ids=enc["input_ids"].to(args.device), attention_mask=enc["attention_mask"].to(args.device))

    print(f"Measuring latency over {args.runs} runs...")
    for i in range(args.runs):
        t = texts[i % len(texts)]
        
        start = time.perf_counter()
        
        # We include tokenization in the time because "per utterance" usually implies end-to-end
        enc = tokenizer(t, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            _ = model(input_ids=enc["input_ids"].to(args.device), attention_mask=enc["attention_mask"].to(args.device))
        
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    p50 = statistics.median(times_ms)
    times_sorted = sorted(times_ms)
    p95 = times_sorted[int(0.95 * len(times_sorted)) - 1]

    print(f"Latency Results (Model: {args.model_name or 'Custom'}):")
    print(f"  p50: {p50:.2f} ms")
    print(f"  p95: {p95:.2f} ms")

if __name__ == "__main__":
    main()