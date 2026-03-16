"""
Run VLM pipeline with Hugging Face token authentication.
Prompts for token if not already set.
"""
import os
import sys
import subprocess
from pathlib import Path

# Check if token is already set
if "HF_TOKEN" not in os.environ:
    print("\n" + "="*60)
    print("HUGGING FACE TOKEN REQUIRED")
    print("="*60)
    print("\nTo access gated models, you need a Hugging Face token.")
    print("Get your token from: https://huggingface.co/settings/tokens")
    print("\nPaste your token below (input will be hidden):")
    print("="*60 + "\n")
    
    # Get token (hidden input)
    import getpass
    token = getpass.getpass("HF Token: ").strip()
    
    if not token:
        print("\n❌ No token provided. Exiting.")
        sys.exit(1)
    
    os.environ["HF_TOKEN"] = token
    print("\n✓ Token set successfully!\n")
else:
    print("\n✓ Using existing HF_TOKEN from environment\n")

# Run the pipeline with BLIP-2 model (lighter, works on CPU)
print("="*60)
print("Starting VLM Pipeline with Salesforce/blip2-flan-t5-xl")
print("="*60)
print("\nThis will take several minutes...")
print("- Loading model: ~1-2 minutes")  
print("- Processing 1 sample: ~1-3 minutes")
print("="*60 + "\n")

# Build command - using BLIP-2 Flan-T5-XL model
cmd = [
    sys.executable,
    "-m", "src.pipeline_llm",
    "--dataset", "dataset",
]

# Run pipeline
result = subprocess.run(cmd, env=os.environ)
sys.exit(result.returncode)
