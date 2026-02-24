import argparse
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from .utils_hebrew import normalize_hebrew


# Cache models globally to avoid reloading
_MODEL_CACHE = {}


def load_blip(device='cpu'):
    key = 'blip'
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
    _MODEL_CACHE[key] = (processor, model)
    if device == 'cuda':
        model.to(device)
    return processor, model


def generate_caption(image_path: str, device='cpu', prefer_hebrew=False):
    processor, model = load_blip(device)
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors='pt')
    if device == 'cuda':
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        model.to('cuda')
    out = model.generate(**inputs, max_new_tokens=64)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def translate_en_to_he(text: str, device='cpu') -> str:
    key = 'en-he-trans'
    if key not in _MODEL_CACHE:
        tok = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-he')
        m = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-he')
        _MODEL_CACHE[key] = (tok, m)
    tok, m = _MODEL_CACHE[key]
    inputs = tok(text, return_tensors='pt', truncation=True)
    if device == 'cuda':
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        m.to('cuda')
    out = m.generate(**inputs, max_new_tokens=100)
    return tok.decode(out[0], skip_special_tokens=True)


def load_embed_model(name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device='cpu'):
    key = f'embed:{name}'
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model = SentenceTransformer(name, device=device)
    _MODEL_CACHE[key] = model
    return model


def embed_texts(texts, model_name=None, device='cpu'):
    model = load_embed_model(model_name, device=device) if model_name else load_embed_model(device=device)
    return model.encode(texts, convert_to_tensor=True)


def compare_texts(text_a: str, text_b: str, model_name=None, device='cpu') -> float:
    a = normalize_hebrew(text_a)
    b = normalize_hebrew(text_b)
    emb = embed_texts([a, b], model_name=model_name, device=device)
    score = util.cos_sim(emb[0], emb[1]).item()
    return score


def process_pair(image_path: Path, answer_text: str, embed_model: str, translate: bool, device: str) -> Tuple[str, float, str]:
    """Generate caption, translate if requested, and compute similarity.

    Returns (caption_he, score, caption_en) for downstream reporting.
    """
    caption_en = generate_caption(str(image_path), device=device)
    caption_he = caption_en
    if translate:
        try:
            caption_he = translate_en_to_he(caption_en, device=device)
        except Exception:
            caption_he = caption_en
    score = compare_texts(caption_he, answer_text, model_name=embed_model, device=device)
    return caption_he, score, caption_en


def main():
    parser = argparse.ArgumentParser(description='Image→text caption + Hebrew similarity pipeline')
    parser.add_argument('--dataset', default='dataset', help='Root dataset folder containing images/ and answers/')
    parser.add_argument('--image', help='Path to image file (single mode)')
    parser.add_argument('--answer', help='Questionnaire answer in Hebrew (single mode)')
    parser.add_argument('--embed-model', default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    parser.add_argument('--no-translate', action='store_true', help='Do not translate captions to Hebrew')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    translate_flag = not args.no_translate

    # Single mode if both provided
    if args.image and args.answer:
        caption_he, score, caption_en = process_pair(Path(args.image), args.answer, args.embed_model, translate_flag, device)
        print('Caption (raw):', caption_en)
        print('Caption (he):', caption_he)
        print('Similarity score:', score)
        return

    # Batch mode on dataset
    dataset_root = Path(args.dataset)
    images_dir = dataset_root / 'images'
    answers_dir = dataset_root / 'answers'
    if not images_dir.exists() or not answers_dir.exists():
        raise SystemExit(f"Dataset folders not found. Expected {images_dir} and {answers_dir}")

    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}])
    if not image_files:
        raise SystemExit('No images found in dataset/images')

    results = []
    for img_path in image_files:
        stem = img_path.stem
        ans_path = answers_dir / f"{stem}.txt"
        if not ans_path.exists():
            print(f"[warn] Missing answer file for {img_path.name} -> {ans_path.name}; skipping")
            continue
        answer_text = ans_path.read_text(encoding='utf-8').strip()
        caption_he, score, caption_en = process_pair(img_path, answer_text, args.embed_model, translate_flag, device)
        results.append((stem, score, caption_he, caption_en))
        print(f"[pair] {stem}: score={score:.4f}")

    if not results:
        print('No pairs processed.')
        return

    avg_score = sum(r[1] for r in results) / len(results)
    print(f"Processed {len(results)} pairs; average similarity={avg_score:.4f}")
    for stem, score, caption_he, caption_en in results:
        print(f"  {stem}: score={score:.4f} | caption_he={caption_he} | caption_en={caption_en}")


if __name__ == '__main__':
    main()
