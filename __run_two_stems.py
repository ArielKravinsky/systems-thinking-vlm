import json
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

TARGETS = {"5_2_52", "10_1_11"}

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = SystemsThinkingDataset(Path("dataset"))

selected = [s for s in dataset if s["stem"] in TARGETS]
selected.sort(key=lambda x: x["stem"])

vlm, processor = load_vlm(VLM_MODEL, device)
embed_model = load_embed(EMBED_MODEL, device)
he_en_tokenizer, he_en_model = load_translation_model(HE_EN_MODEL, device)
en_he_tokenizer, en_he_model = load_translation_model(EN_HE_MODEL, device)

results = []
for i, sample in enumerate(selected, 1):
    t0 = time.time()
    llm_answer = ask_vlm(
        vlm,
        processor,
        sample["image"],
        sample["question"],
        he_en_tokenizer,
        he_en_model,
        en_he_tokenizer,
        en_he_model,
        device,
    )
    score = compute_similarity(embed_model, llm_answer, sample["answer"])
    elapsed = round(time.time() - t0, 2)
    row = {
        "stem": sample["stem"],
        "similarity": score,
        "similarity_percent": round(max(0.0, min(1.0, score)) * 100, 2),
        "elapsed_sec": elapsed,
        "subject_answer": sample["answer"],
        "llm_answer": llm_answer,
    }
    results.append(row)
    print(f"[{i}/{len(selected)}] {sample['stem']} | {row['similarity_percent']:.2f}% | {elapsed}s")

out = Path("vlm_results_last2.json")
out.write_text(json.dumps({"device": device, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"WROTE {out}")
