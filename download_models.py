"""
Download and cache models before running the pipeline.
This script downloads the VLM and embedding models with progress indicators.
"""
import argparse
from pathlib import Path
import os
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Default models
VLM_MODEL = "Salesforce/blip2-opt-2.7b"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def check_model_exists(model_name: str, cache_dir: str = None) -> bool:
    """Check if model is already cached locally."""
    # Get cache directory
    if cache_dir is None:
        cache_dir = os.getenv('TRANSFORMERS_CACHE', 
                             os.getenv('HF_HOME', 
                                      os.path.join(Path.home(), '.cache', 'huggingface')))
    
    cache_path = Path(cache_dir)
    
    # Check if model directory exists in cache
    # Transformers cache structure: models--{org}--{model}
    model_slug = model_name.replace('/', '--')
    model_cache = cache_path / 'hub' / f'models--{model_slug}'
    
    if model_cache.exists():
        # Check if it has the required files
        snapshots_dir = model_cache / 'snapshots'
        if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
            return True
    
    return False


def download_vlm(model_name: str, force: bool = False):
    """Download VLM model and processor with progress."""
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    
    # Check if already downloaded
    if not force and check_model_exists(model_name):
        print(f"\n{'='*60}")
        print(f"VLM Model: {model_name}")
        print(f"{'='*60}")
        print("✓ Model already cached locally - skipping download")
        print("  Use --force to re-download")
        print(f"{'='*60}\n")
        return
    
    print(f"\n{'='*60}")
    print(f"Downloading VLM Model: {model_name}")
    print(f"{'='*60}")
    print("This is BLIP-2 2.7B model (~5GB) and may take 3-10 minutes...")
    print("Progress will be shown during download.\n")
    
    # Download model
    print("Downloading model weights...")
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu"  # Download only, don't load to device
    )
    print("✓ Model weights downloaded successfully!")
    
    # Download processor
    print("\nDownloading processor and tokenizer...")
    processor = Blip2Processor.from_pretrained(model_name)
    print("✓ Processor downloaded successfully!")
    
    # Clean up from memory
    del model
    del processor
    
    print(f"\n{'='*60}")
    print("VLM Model download complete!")
    print(f"{'='*60}\n")


def download_embedding_model(model_name: str, force: bool = False):
    """Download embedding model with progress."""
    # Check if already downloaded
    if not force and check_model_exists(model_name):
        print(f"\n{'='*60}")
        print(f"Embedding Model: {model_name}")
        print(f"{'='*60}")
        print("✓ Model already cached locally - skipping download")
        print("  Use --force to re-download")
        print(f"{'='*60}\n")
        return
    
    print(f"\n{'='*60}")
    print(f"Downloading Embedding Model: {model_name}")
    print(f"{'='*60}\n")
    
    # Download model
    print("Downloading model...")
    model = SentenceTransformer(model_name, device="cpu")
    print("✓ Embedding model downloaded successfully!")
    
    # Clean up from memory
    del model
    
    print(f"\n{'='*60}")
    print("Embedding model download complete!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Download models for VLM pipeline")
    parser.add_argument(
        "--vlm", 
        default=VLM_MODEL, 
        help=f"VLM model to download (default: {VLM_MODEL})"
    )
    parser.add_argument(
        "--embed", 
        default=EMBED_MODEL, 
        help=f"Embedding model to download (default: {EMBED_MODEL})"
    )
    parser.add_argument(
        "--skip-vlm", 
        action="store_true", 
        help="Skip VLM download"
    )
    parser.add_argument(
        "--skip-embed", 
        action="store_true", 
        help="Skip embedding model download"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if models are cached"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("MODEL DOWNLOAD SCRIPT")
    print("="*60)
    print(f"VLM Model: {args.vlm}")
    print(f"Embedding Model: {args.embed}")
    print("="*60 + "\n")
    
    # Download VLM
    if not args.skip_vlm:
        try:
            download_vlm(args.vlm, force=args.force)
        except Exception as e:
            print(f"\n❌ Error downloading VLM: {e}")
            return 1
    else:
        print("Skipping VLM download (--skip-vlm flag set)\n")
    
    # Download embedding model
    if not args.skip_embed:
        try:
            download_embedding_model(args.embed, force=args.force)
        except Exception as e:
            print(f"\n❌ Error downloading embedding model: {e}")
            return 1
    else:
        print("Skipping embedding model download (--skip-embed flag set)\n")
    
    print("\n" + "="*60)
    print("✓ ALL MODELS DOWNLOADED SUCCESSFULLY!")
    print("="*60)
    print("\nYou can now run the pipeline:")
    print("  python -m src.pipeline_llm --dataset dataset --output vlm_results.json")
    print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
