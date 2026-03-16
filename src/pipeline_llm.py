import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from tqdm import tqdm
from bert_score import BERTScorer
from .utils_hebrew import normalize_hebrew
from .dataset_loader import SystemsThinkingDataset

CSV_COLUMNS = [
    "stem",
    "question_num",
    "answer_num",
    "participant_num",
    "image_path",
    "question_he",
    "question_en",
    "full_prompt",
    "vlm_raw_answer_en",
    "used_fallback",
    "fallback_prompt",
    "vlm_answer_en",
    "vlm_answer_he",
    "subject_answer_he",
    "similarity",
    "similarity_percent",
    "comparison_method",
    "subject_answer_en",
    "similarity_labse_he",
    "similarity_labse_he_percent",
    "similarity_en",
    "similarity_en_percent",
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
    "vlm_model",
    "embed_model",
    "he_en_model",
    "en_he_model",
    "device",
    "timestamp",
]

# Default models
VLM_MODEL = "Salesforce/blip2-flan-t5-xl"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LABSE_MODEL = "sentence-transformers/LaBSE"
EN_HE_MODEL = "Helsinki-NLP/opus-mt-en-he"
HE_EN_MODEL = "Helsinki-NLP/opus-mt-tc-big-he-en"

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def load_vlm(model_name: str, device: str):
    """Load VLM model from cache (must be downloaded first)."""
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    print(f"Loading VLM model: {model_name}...")
    print("This may take 1-2 minutes on CPU...")

    import gc
    gc.collect()

    if device == "cuda":
        vlm = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        print("Loading BLIP-2 model on CPU...")
        vlm = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )

    processor = Blip2Processor.from_pretrained(model_name)
    print("[OK] VLM loaded successfully!")
    return vlm, processor


def load_embed(model_name: str, device: str):
    """Load embedding model from cache (must be downloaded first)."""
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name, device=device)
    print("[OK] Embedding model loaded!")
    return model


def load_translation_model(model_name: str, device: str):
    """Load a translation model/tokenizer once and reuse across samples."""
    print(f"Loading translation model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device == "cuda":
        model = model.to(device)
    print("[OK] Translation model loaded!")
    return tokenizer, model


def translate_text(text: str, tokenizer, model, device: str) -> str:
    """Translate text using a preloaded Marian model."""
    if not isinstance(text, str) or not text.strip():
        return ""

    inputs = tokenizer(text, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def _deduplicate_sentences(text: str) -> str:
    """Remove repeated sentences from a looping model output."""
    # Split on sentence boundaries, preserve trailing period
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    seen = []
    for s in sentences:
        s_norm = s.strip().lower()
        if s_norm and s_norm not in seen:
            seen.append(s_norm)
    # Reconstruct using original-case first occurrences
    unique = []
    seen_set = set()
    for s in sentences:
        key = s.strip().lower()
        if key and key not in seen_set:
            seen_set.add(key)
            unique.append(s.strip())
    return " ".join(unique)


def _is_degenerate_answer(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t or len(t) < 3:
        return True
    if "\ufffd" in t:
        return True
    if "no, no, no" in t:
        return True
    # Word-level repetition check
    words = re.findall(r"[a-zA-Z']+", t)
    if len(words) >= 4:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.4:
            return True
    # Sentence-level loop: first sentence re-appears verbatim later
    sentences = re.split(r'(?<=[.!?])\s+', t)
    if len(sentences) >= 2:
        first = sentences[0].strip()
        if first and any(sentences[i].strip() == first for i in range(1, len(sentences))):
            return True
    return False


def _extract_concept(question_en: str) -> str:
    """Pull the quoted concept sentence out of questions like:
    'Explain why you chose this image as reflecting the sentence "X"'
    Falls back to the full question if no quoted text is found.
    """
    m = re.search(r'["\u201c\u201d\u2018\u2019]([^"\u201c\u201d\u2018\u2019]+)["\u201c\u201d\u2018\u2019]', question_en)
    return m.group(1).strip() if m else question_en


def ask_vlm(
    vlm,
    processor,
    pil_image: Image.Image,
    question_he: str,
    he_en_tokenizer,
    he_en_model,
    en_he_tokenizer,
    en_he_model,
    device: str,
) -> dict:
    """Ask VLM a question about an image.

    Returns a trace dict with every intermediate value:
        question_en, full_prompt, vlm_raw_answer_en, used_fallback,
        fallback_prompt, vlm_answer_en, vlm_answer_he
    """
    img = pil_image.copy()
    img.thumbnail((640, 640), Image.Resampling.LANCZOS)

    question_en = translate_text(question_he, he_en_tokenizer, he_en_model, device)
    question_en = re.sub(r"\s+", " ", question_en).strip()

    # Original direct question-answer prompt: give the model the Hebrew question
    # and ask it to answer in English with a systems-thinking lens.
    prompt = (
        "You are an expert in systems thinking. "
        "Based on the image, answer the user's question in English. "
        "Focus on relationships, interactions, or decision dynamics when relevant.\n"
        f"Question (Hebrew): {question_he}\n"
        "Answer:"
    )

    is_enc_dec = getattr(vlm.config, "is_encoder_decoder", False)

    inputs = processor(images=img, text=prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    n_input = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = vlm.generate(
            **inputs,
            max_new_tokens=200,
            min_new_tokens=8,
            do_sample=False,
            repetition_penalty=1.5,
        )

    # Encoder-decoder (Flan-T5): out[0] is purely the generated tokens.
    # Decoder-only (OPT): out[0] echoes the input — slice it off.
    raw_tokens = out[0] if is_enc_dec else out[0][n_input:]
    decoded = _deduplicate_sentences(
        processor.decode(raw_tokens, skip_special_tokens=True).strip()
    )
    vlm_raw_answer_en = decoded
    answer_en = vlm_raw_answer_en
    used_fallback = False
    fallback_prompt_used = ""

    if _is_degenerate_answer(answer_en):
        used_fallback = True
        fallback_prompt_used = (
            "Describe what concrete objects, actions, or relationships are visible "
            "in this image, and what role they might play in a systems thinking context."
        )
        fallback_inputs = processor(images=img, text=fallback_prompt_used, return_tensors="pt")
        if device == "cuda":
            fallback_inputs = {k: v.to(device) for k, v in fallback_inputs.items()}
        n_fallback_input = fallback_inputs["input_ids"].shape[1]
        with torch.no_grad():
            fallback_out = vlm.generate(
                **fallback_inputs,
                max_new_tokens=200,
                min_new_tokens=8,
                do_sample=False,
                repetition_penalty=1.5,
            )
        raw_fallback = fallback_out[0] if is_enc_dec else fallback_out[0][n_fallback_input:]
        fb_decoded = _deduplicate_sentences(
            processor.decode(raw_fallback, skip_special_tokens=True).strip()
        )
        answer_en = fb_decoded

    if _is_degenerate_answer(answer_en):
        answer_en = "The image shows people interacting, and their interactions can influence later outcomes."

    answer_he = translate_text(answer_en, en_he_tokenizer, en_he_model, device)

    return {
        "question_en": question_en,
        "full_prompt": prompt,
        "vlm_raw_answer_en": vlm_raw_answer_en,
        "used_fallback": used_fallback,
        "fallback_prompt": fallback_prompt_used,
        "vlm_answer_en": answer_en,
        "vlm_answer_he": answer_he,
    }


def compute_similarity(embed_model, text_a: str, text_b: str) -> float:
    a = normalize_hebrew(text_a)
    b = normalize_hebrew(text_b)
    emb = embed_model.encode([a, b], convert_to_tensor=True)
    return util.cos_sim(emb[0], emb[1]).item()


def compute_bertscore(scorer: "BERTScorer", hyp: str, ref: str) -> tuple:
    """Returns (precision, recall, f1) floats. Falls back to 0s on error."""
    if not hyp.strip() or not ref.strip():
        return 0.0, 0.0, 0.0
    try:
        P, R, F = scorer.score([hyp], [ref])
        return round(P[0].item(), 4), round(R[0].item(), 4), round(F[0].item(), 4)
    except Exception:
        return 0.0, 0.0, 0.0


def process_dataset(dataset_root: Path, vlm_name: str, embed_name: str, device: str, output_path: Path, csv_path: Path):
    """Process dataset using SystemsThinkingDataset loader. Saves JSON + CSV incrementally."""
    print(f"\nInitializing dataset from: {dataset_root}")
    dataset = SystemsThinkingDataset(dataset_root)

    print(f"Found {len(dataset)} samples in dataset\n")

    vlm, processor = load_vlm(vlm_name, device)
    embed_model = load_embed(embed_name, device)
    labse_model = load_embed(LABSE_MODEL, device)
    he_en_tokenizer, he_en_model = load_translation_model(HE_EN_MODEL, device)
    en_he_tokenizer, en_he_model = load_translation_model(EN_HE_MODEL, device)
    print("Loading BERTScorer (roberta-large, English)...")
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=False, device="cpu")
    print("[OK] BERTScorer loaded!")

    results = []
    payload_meta = {"device": device, "vlm": vlm_name, "embed": embed_name,
                    "he_en_model": HE_EN_MODEL, "en_he_model": EN_HE_MODEL}

    # Open CSV and write header once
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    print(f"\nProcessing {len(dataset)} samples...\n")
    try:
        for sample in tqdm(dataset, total=len(dataset), desc="Processing samples", unit="sample"):
            question_num = sample["question_num"]
            answer_num = sample["answer_num"]
            participant_num = sample["participant_num"]
            pil_image = sample["image"]
            question = sample["question"]
            subject_answer = sample["answer"]
            image_path = str(sample["image_path"])
            stem = f"{question_num}_{answer_num}_{participant_num}"
            ts = datetime.utcnow().isoformat(timespec="seconds")

            trace = ask_vlm(
                vlm,
                processor,
                pil_image,
                question,
                he_en_tokenizer,
                he_en_model,
                en_he_tokenizer,
                en_he_model,
                device,
            )

            # translate human answer to English for En-En comparisons
            subject_answer_en = translate_text(subject_answer, he_en_tokenizer, he_en_model, device)

            # 1. mpnet cosine similarity (He-He) — original metric
            score = compute_similarity(embed_model, trace["vlm_answer_he"], subject_answer)
            similarity_pct = round(max(0.0, min(1.0, score)) * 100, 2)

            # 2. LaBSE cosine similarity (He-He)
            score_labse = compute_similarity(labse_model, trace["vlm_answer_he"], subject_answer)
            similarity_labse_pct = round(max(0.0, min(1.0, score_labse)) * 100, 2)

            # 3. mpnet cosine similarity (En-En)
            score_en = compute_similarity(embed_model, trace["vlm_answer_en"], subject_answer_en)
            similarity_en_pct = round(max(0.0, min(1.0, score_en)) * 100, 2)

            # 4. BERTScore (En-En)
            bs_p, bs_r, bs_f1 = compute_bertscore(bert_scorer, trace["vlm_answer_en"], subject_answer_en)

            # ── JSON result ──────────────────────────────────────────────────
            result = {
                "stem": stem,
                "question_num": question_num,
                "answer_num": answer_num,
                "participant_num": participant_num,
                "image_path": image_path,
                "question_he": question,
                "question_en": trace["question_en"],
                "full_prompt": trace["full_prompt"],
                "vlm_raw_answer_en": trace["vlm_raw_answer_en"],
                "used_fallback": trace["used_fallback"],
                "fallback_prompt": trace["fallback_prompt"],
                "vlm_answer_en": trace["vlm_answer_en"],
                "vlm_answer_he": trace["vlm_answer_he"],
                "subject_answer_he": subject_answer,
                "similarity": score,
                "similarity_percent": similarity_pct,
                "comparison_method": "multilingual semantic embedding cosine similarity",
                "subject_answer_en": subject_answer_en,
                "similarity_labse_he": score_labse,
                "similarity_labse_he_percent": similarity_labse_pct,
                "similarity_en": score_en,
                "similarity_en_percent": similarity_en_pct,
                "bertscore_precision": bs_p,
                "bertscore_recall": bs_r,
                "bertscore_f1": bs_f1,
                "timestamp": ts,
            }
            results.append(result)

            # ── CSV row ───────────────────────────────────────────────────────
            writer.writerow({
                "stem": stem,
                "question_num": question_num,
                "answer_num": answer_num,
                "participant_num": participant_num,
                "image_path": image_path,
                "question_he": question,
                "question_en": trace["question_en"],
                "full_prompt": trace["full_prompt"],
                "vlm_raw_answer_en": trace["vlm_raw_answer_en"],
                "used_fallback": trace["used_fallback"],
                "fallback_prompt": trace["fallback_prompt"],
                "vlm_answer_en": trace["vlm_answer_en"],
                "vlm_answer_he": trace["vlm_answer_he"],
                "subject_answer_he": subject_answer,
                "similarity": score,
                "similarity_percent": similarity_pct,
                "comparison_method": "multilingual semantic embedding cosine similarity",
                "subject_answer_en": subject_answer_en,
                "similarity_labse_he": score_labse,
                "similarity_labse_he_percent": similarity_labse_pct,
                "similarity_en": score_en,
                "similarity_en_percent": similarity_en_pct,
                "bertscore_precision": bs_p,
                "bertscore_recall": bs_r,
                "bertscore_f1": bs_f1,
                "vlm_model": vlm_name,
                "embed_model": embed_name,
                "he_en_model": HE_EN_MODEL,
                "en_he_model": EN_HE_MODEL,
                "device": device,
                "timestamp": ts,
            })
            csv_file.flush()

            tqdm.write(f"{stem}: {score:.4f}")

            # ── JSON incremental save ─────────────────────────────────────────
            output_path.write_text(
                json.dumps({**payload_meta, "results": results}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    finally:
        csv_file.close()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="VLM question answering vs subject answers",
        epilog="Note: Run download_models.py first to download required models.",
    )
    parser.add_argument("--dataset", default="dataset", help="dataset root with images/, answers/, questions/")
    parser.add_argument("--vlm", default=VLM_MODEL, help="vision-language model name")
    parser.add_argument("--embed-model", default=EMBED_MODEL, help="embedding model for similarity")
    parser.add_argument("--output", default=None,
                        help="output JSON file (default: vlm_results_<YYYYMMDD_HHMMSS>.json)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    print("\nIf you see 'local_files_only' errors, run: python download_models.py\n")

    # Auto-stamp filename so every run produces unique files
    run_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else Path(f"results/vlm_results_{run_ts}.json")
    csv_path = output_path.with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Output files: {output_path}  |  {csv_path}\n")

    results = process_dataset(Path(args.dataset), args.vlm, args.embed_model, device, output_path, csv_path)
    print(f"Processed {len(results)} pairs; wrote {output_path} and {csv_path}")


if __name__ == "__main__":
    main()
