import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

print("Loading VLM model - this may take 2-5 minutes...")
print("Model: llava-hf/llava-1.5-7b-hf")
print("Device: CPU (float32)")
print("")

vlm = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)

print("\n✓ Model loaded successfully!")

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
print("✓ Processor loaded successfully!")

# Test inference
from PIL import Image
import requests
from io import BytesIO

print("\nTesting inference with sample image...")
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

prompt = "USER: <image>\nWhat is in this image?\nASSISTANT:"
inputs = processor(images=image, text=prompt, return_tensors="pt")

print("Generating response...")
with torch.no_grad():
    output = vlm.generate(**inputs, max_new_tokens=50)

result = processor.batch_decode(output, skip_special_tokens=True)[0]
print(f"\nVLM Response: {result}")
print("\n✓ Inference test successful!")
