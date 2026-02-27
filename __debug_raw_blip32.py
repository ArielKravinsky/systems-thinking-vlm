import torch
from pathlib import Path
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from src.dataset_loader import SystemsThinkingDataset

model_name = "Salesforce/blip2-opt-2.7b"
device = "cpu"

print("Loading BLIP2 float32 on CPU...")
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    device_map="cpu",
)
processor = Blip2Processor.from_pretrained(model_name)
print("loaded")

dataset = SystemsThinkingDataset(Path("dataset"))
sample = dataset[0]
img = sample["image"].copy()
img.thumbnail((640, 640), Image.Resampling.LANCZOS)

prompt = "Describe the image in one short English sentence."
inputs = processor(images=img, text=prompt, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=48, do_sample=False, min_new_tokens=8)
raw = processor.decode(out[0], skip_special_tokens=True)
print("RAW:", raw)
