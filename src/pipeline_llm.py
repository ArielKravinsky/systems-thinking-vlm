import argparse
import json
from pathlib import Path
from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from tqdm import tqdm
from .utils_hebrew import normalize_hebrew
from .dataset_loader import SystemsThinkingDataset

# Default models
VLM_MODEL = "Salesforce/blip2-opt-2.7b"  # Lighter model, works on CPU
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EN_HE_MODEL = "Helsinki-NLP/opus-mt-en-he"

# Set environment variable to avoid memory issues
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'


def load_vlm(model_name: str, device: str):
    """Load VLM model from cache (must be downloaded first)."""
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    
    print(f"Loading VLM model: {model_name}...")
    print("This may take 1-2 minutes on CPU...")
    
    import gc
    gc.collect()  # Clear memory before loading
    
    if device == "cuda":
        vlm = Blip2ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        # CPU: BLIP-2 is much lighter than LLaVA
        print("Loading BLIP-2 model on CPU...")
        vlm = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
    
    processor = Blip2Processor.from_pretrained(model_name)
    print("✓ VLM loaded successfully!")
    return vlm, processor


def load_embed(model_name: str, device: str):
    """Load embedding model from cache (must be downloaded first)."""
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name, device=device)
    print("✓ Embedding model loaded!")
    return model


def load_translation_model(model_name: str, device: str):
    """Load a translation model/tokenizer once and reuse across samples."""
    print(f"Loading translation model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device == "cuda":
        model = model.to(device)
    print("✓ Translation model loaded!")
    return tokenizer, model


def translate_text(text: str, tokenizer, model, device: str) -> str:
    """Translate text using a preloaded Marian model."""
    if not isinstance(text, str) or not text.strip():
        return ""

    inputs = tokenizer(text, return_tensors="pt", padding=True)
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def ask_vlm(
    vlm,
    processor,
    pil_image: Image.Image,
    question_he: str,
    en_he_tokenizer,
    en_he_model,
    device: str,
) -> str:
    """Ask VLM a question about an image and return a Hebrew expert answer."""
    prompt = (
        "You are an expert in systems thinking. "
        "Based on the image, answer the user's question in one concise sentence in English. "
        "Focus on relationships, interactions, or decision dynamics when relevant.\n"
        f"Question (Hebrew): {question_he}\n"
        "Answer:"
    )
    
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt")
    
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        out = vlm.generate(**inputs, max_new_tokens=128, do_sample=False)
    
    answer_en = processor.decode(out[0], skip_special_tokens=True).strip()
    
    # Clean up the prompt from the answer
    if "Answer:" in answer_en:
        answer_en = answer_en.split("Answer:")[-1].strip()
    if prompt in answer_en:
        answer_en = answer_en.replace(prompt, "").strip()
    
    # Translate English answer to Hebrew
    answer_he = translate_text(answer_en, en_he_tokenizer, en_he_model, device)
    return answer_he


def compute_similarity(embed_model, text_a: str, text_b: str):
    a = normalize_hebrew(text_a)
    b = normalize_hebrew(text_b)
    emb = embed_model.encode([a, b], convert_to_tensor=True)
    return util.cos_sim(emb[0], emb[1]).item()


def process_dataset(dataset_root: Path, vlm_name: str, embed_name: str, device: str):
    """Process dataset using SystemsThinkingDataset loader."""
    print(f"\nInitializing dataset from: {dataset_root}")
    dataset = SystemsThinkingDataset(dataset_root)
    
    print(f"Found {len(dataset)} samples in dataset\n")
    
    vlm, processor = load_vlm(vlm_name, device)
    embed_model = load_embed(embed_name, device)
    en_he_tokenizer, en_he_model = load_translation_model(EN_HE_MODEL, device)
    
    results = []
    
    print(f"\nProcessing {len(dataset)} samples...\n")
    for sample in tqdm(dataset, desc="Processing samples", unit="sample"):
        question_num = sample['question_num']
        answer_num = sample['answer_num']
        participant_num = sample['participant_num']
        pil_image = sample['image']
        question = sample['question']
        subject_answer = sample['answer']
        
        # Ask VLM the question about the image
        vlm_answer = ask_vlm(
            vlm,
            processor,
            pil_image,
            question,
            en_he_tokenizer,
            en_he_model,
            device,
        )
        
        # Compute similarity
        score = compute_similarity(embed_model, vlm_answer, subject_answer)
        
        result = {
            "stem": f"{question_num}_{answer_num}_{participant_num}",
            "question_num": question_num,
            "answer_num": answer_num,
            "participant_num": participant_num,
            "question": question,
            "subject_answer": subject_answer,
            "llm_answer": vlm_answer,
            "similarity": score,
            "similarity_percent": round(max(0.0, min(1.0, score)) * 100, 2),
            "comparison_method": "multilingual semantic embedding cosine similarity"
        }
        results.append(result)
        tqdm.write(f"Q{question_num}_A{answer_num}_P{participant_num}: {score:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="VLM question answering vs subject answers",
        epilog="Note: Run download_models.py first to download required models."
    )
    parser.add_argument("--dataset", default="dataset", help="dataset root with images/, answers/, questions/")
    parser.add_argument("--vlm", default=VLM_MODEL, help="vision-language model name")
    parser.add_argument("--embed-model", default=EMBED_MODEL, help="embedding model for similarity")
    parser.add_argument("--output", default="vlm_results.json", help="output JSON file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    print("\nIf you see 'local_files_only' errors, run: python download_models.py\n")
    
    results = process_dataset(Path(args.dataset), args.vlm, args.embed_model, device)
    payload = {"device": device, "vlm": args.vlm, "embed": args.embed_model, "results": results}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Processed {len(results)} pairs; wrote {args.output}")


if __name__ == "__main__":
    main()
