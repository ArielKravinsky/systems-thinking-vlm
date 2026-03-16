"""Simple test to verify model loading"""
import sys
print("Starting model loading test...")

print("\n1. Importing torch...")
import torch
print(f"   ✓ Torch version: {torch.__version__}")
print(f"   ✓ CUDA available: {torch.cuda.is_available()}")

print("\n2. Importing transformers...")
from transformers import LlavaForConditionalGeneration, AutoProcessor
print("   ✓ Transformers imported")

print("\n3. Importing sentence-transformers...")
from sentence_transformers import SentenceTransformer
print("   ✓ Sentence-transformers imported")

print("\n4. Loading embedding model...")
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device="cpu")
print("   ✓ Embedding model loaded")

print("\n5. Testing embedding...")
test_emb = embed_model.encode("test", convert_to_tensor=True)
print(f"   ✓ Embedding shape: {test_emb.shape}")

print("\n✓ All imports and basic loading successful!")
print("\nNow attempting VLM model load (this will take 2-3 minutes)...")

print("\nLoading LLaVA model...")
vlm = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)
print("✓ VLM model loaded!")

print("\nLoading VLM processor...")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
print("✓ VLM processor loaded!")

print("\n" + "="*60)
print("✓✓✓ ALL MODELS LOADED SUCCESSFULLY! ✓✓✓")
print("="*60)
