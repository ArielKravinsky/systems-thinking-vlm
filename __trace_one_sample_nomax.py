from pathlib import Path
import torch
from PIL import Image
from src.dataset_loader import SystemsThinkingDataset
from src.pipeline_llm import load_vlm, load_translation_model, EN_HE_MODEL, VLM_MODEL

device = "cpu"
ds = SystemsThinkingDataset(Path("dataset"))
s = ds[0]

vlm, processor = load_vlm(VLM_MODEL, device)
tok, tr = load_translation_model(EN_HE_MODEL, device)

img = s["image"].copy()
img.thumbnail((640, 640), Image.Resampling.LANCZOS)

prompt = "Describe the image in one short English sentence."
inputs = processor(images=img, text=prompt, return_tensors="pt")

with torch.no_grad():
    out = vlm.generate(
        **inputs,
        max_new_tokens=48,
        min_new_tokens=8,
        do_sample=False,
    )
raw = processor.decode(out[0], skip_special_tokens=True).strip()

en = raw.replace(prompt, "").strip()
tr_in = en if en else "The image shows people interacting in context."
tr_inputs = tok(tr_in, return_tensors="pt", padding=True)
with torch.no_grad():
    tr_out = tr.generate(**tr_inputs, max_new_tokens=128)
he = tok.decode(tr_out[0], skip_special_tokens=True).strip()

print("STEM:", s["stem"])
print("RAW:", raw)
print("EN:", en)
print("HE:", he)
