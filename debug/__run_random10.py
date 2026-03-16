import json
import random
import time
from pathlib import Path
import torch

from src.dataset_loader import SystemsThinkingDataset
from src.pipeline_llm import (
    load_vlm,
    load_embed,
    load_translation_model,
    ask_vlm,
    compute_similarity,
    VLM_MODEL,
    EMBED_MODEL,
    HE_EN_MODEL,
    EN_HE_MODEL,
)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}", flush=True)
dataset = SystemsThinkingDataset(Path("dataset"))
print(f"Valid samples available: {len(dataset)}", flush=True)

n = min(10, len(dataset))
indices = random.sample(range(len(dataset)), n)
print(f"Running on random sample count: {n}", flush=True)

vlm, processor = load_vlm(VLM_MODEL, device)
embed_model = load_embed(EMBED_MODEL, device)
he_en_tokenizer, he_en_model = load_translation_model(HE_EN_MODEL, device)
en_he_tokenizer, en_he_model = load_translation_model(EN_HE_MODEL, device)

results = []
for k, idx in enumerate(indices, start=1):
    sample = dataset[idx]
    stem = sample['stem']
    t0 = time.time()
    print(f"[{k}/{n}] START {stem}", flush=True)

    llm_answer = ask_vlm(
        vlm,
        processor,
        sample['image'],
        sample['question'],
        he_en_tokenizer,
        he_en_model,
        en_he_tokenizer,
        en_he_model,
        device,
    )
    score = compute_similarity(embed_model, llm_answer, sample['answer'])
    percent = round(max(0.0, min(1.0, score)) * 100, 2)
    dt = time.time() - t0

    results.append({
        "stem": stem,
        "similarity": score,
        "similarity_percent": percent,
        "elapsed_sec": round(dt, 2),
        "subject_answer": sample['answer'],
        "llm_answer": llm_answer,
    })
    print(f"[{k}/{n}] DONE  {stem} | similarity={score:.4f} | percent={percent:.2f}% | time={dt:.1f}s", flush=True)

out = Path("vlm_results_random10.json")
out.write_text(json.dumps({"device": device, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"DONE: wrote {out.as_posix()} with {len(results)} rows", flush=True)
