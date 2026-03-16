from pathlib import Path
import torch
from PIL import Image
from src.dataset_loader import SystemsThinkingDataset
from src.pipeline_llm import load_vlm, load_translation_model, EN_HE_MODEL, VLM_MODEL

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = SystemsThinkingDataset(Path("dataset"))
sample = dataset[0]

vlm, processor = load_vlm(VLM_MODEL, device)
tok, tr = load_translation_model(EN_HE_MODEL, device)

img = sample["image"].copy()
img.thumbnail((640, 640), Image.Resampling.LANCZOS)

prompts = [
    "Describe the image in one short English sentence.",
    "You are an expert in systems thinking. Describe actors and relationships visible in the image in one concise English sentence.",
    f"Question (Hebrew): {sample['question']}\nAnswer in one English sentence:",
]

for i, prompt in enumerate(prompts, 1):
    inputs = processor(images=img, text=prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = vlm.generate(**inputs, max_new_tokens=48, do_sample=False, max_time=25)
    raw = processor.decode(out[0], skip_special_tokens=True).strip()

    en = raw
    if "Answer:" in en:
        en = en.split("Answer:")[-1].strip()
    if prompt in en:
        en = en.replace(prompt, "").strip()

    tr_in = en if en else "The image shows people interacting in context."
    tr_inputs = tok(tr_in, return_tensors="pt", padding=True)
    if device == "cuda":
        tr_inputs = {k: v.to(device) for k, v in tr_inputs.items()}
    with torch.no_grad():
        tr_out = tr.generate(**tr_inputs, max_new_tokens=128)
    he = tok.decode(tr_out[0], skip_special_tokens=True).strip()

    print(f"\n--- Prompt {i} ---")
    print("PROMPT:", prompt)
    print("RAW:", raw)
    print("CLEAN_EN:", en)
    print("HE:", he)
