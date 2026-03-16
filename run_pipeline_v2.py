"""
run_pipeline_v2.py
==================
Pipeline v2: dimension-classified, concept-tailored prompts.

Identical output format to pipeline v1 (vlm_results_*.csv / .json) plus
two extra columns:
  dimension_id         – which systems-thinking dimension was matched
  dimension_confidence – cosine similarity of concept to winning anchor

The prompt sent to the VLM is no longer hardcoded.  Instead:
  1. Extract the concept from the question  (same as v1)
  2. classify_dimension() embeds the concept and cosine-matches it against
     ~50 anchor phrases across 9 dimensions.
  3. The winning dimension's template is used.
  4. If confidence < 0.50, falls back to the v1 fill-in-blank prompt.

Usage:
    conda run -n systems_thinking python run_pipeline_v2.py
    conda run -n systems_thinking python run_pipeline_v2.py --device cuda
"""

import argparse
import csv
import json
import os
import re
import sys
import gc
from datetime import datetime
from pathlib import Path

# Allow importing from src/ when run from repo root
sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from tqdm import tqdm
from bert_score import BERTScorer

from src.utils_hebrew import normalize_hebrew
from src.dataset_loader import SystemsThinkingDataset
from src.prompt_dimensions import classify_dimension

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# ── models ────────────────────────────────────────────────────────────────────
VLM_MODEL   = "Salesforce/blip2-flan-t5-xl"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LABSE_MODEL = "sentence-transformers/LaBSE"
EN_HE_MODEL = "Helsinki-NLP/opus-mt-en-he"
HE_EN_MODEL = "Helsinki-NLP/opus-mt-tc-big-he-en"

# ── CSV columns (v1 columns preserved + 2 new) ────────────────────────────────
CSV_COLUMNS = [
    "stem", "question_num", "answer_num", "participant_num", "image_path",
    "question_he", "question_en",
    "dimension_id", "dimension_confidence",          # ← new in v2
    "full_prompt",
    "vlm_raw_answer_en", "used_fallback", "fallback_prompt",
    "vlm_answer_en", "vlm_answer_he",
    "subject_answer_he",
    "similarity", "similarity_percent", "comparison_method",
    "subject_answer_en",
    "similarity_labse_he", "similarity_labse_he_percent",
    "similarity_en", "similarity_en_percent",
    "bertscore_precision", "bertscore_recall", "bertscore_f1",
    "vlm_model", "embed_model", "he_en_model", "en_he_model", "device", "timestamp",
]


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADERS  (identical to v1)
# ══════════════════════════════════════════════════════════════════════════════

def load_vlm(model_name: str, device: str):
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    print(f"Loading VLM: {model_name} ...")
    gc.collect()
    if device == "cuda":
        vlm = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto")
    else:
        vlm = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float32,
            low_cpu_mem_usage=True, device_map="cpu")
    processor = Blip2Processor.from_pretrained(model_name)
    print("[OK] VLM loaded")
    return vlm, processor


def load_embed(model_name: str, device: str) -> SentenceTransformer:
    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name, device=device)
    print("[OK] Embedding model loaded")
    return model


def load_translation_model(model_name: str, device: str):
    print(f"Loading translation model: {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device == "cuda":
        mdl = mdl.to(device)
    print("[OK] Translation model loaded")
    return tok, mdl


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT HELPERS  (identical to v1)
# ══════════════════════════════════════════════════════════════════════════════

def translate_text(text: str, tokenizer, model, device: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    inputs = tokenizer(text, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def _deduplicate_sentences(text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    seen_set, unique = set(), []
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
    words = re.findall(r"[a-zA-Z']+", t)
    if len(words) >= 4 and len(set(words)) / len(words) < 0.4:
        return True
    sentences = re.split(r'(?<=[.!?])\s+', t)
    if len(sentences) >= 2:
        first = sentences[0].strip()
        if first and any(sentences[i].strip() == first for i in range(1, len(sentences))):
            return True
    return False


def _extract_concept(question_en: str) -> str:
    m = re.search(
        r'["\u201c\u201d\u2018\u2019]([^"\u201c\u201d\u2018\u2019]+)["\u201c\u201d\u2018\u2019]',
        question_en,
    )
    return m.group(1).strip() if m else question_en


# ══════════════════════════════════════════════════════════════════════════════
#  VLM INFERENCE  — updated to use dimension classifier
# ══════════════════════════════════════════════════════════════════════════════

def ask_vlm(
    vlm, processor,
    pil_image: Image.Image,
    question_he: str,
    he_en_tokenizer, he_en_model,
    en_he_tokenizer, en_he_model,
    embed_model: SentenceTransformer,      # ← used for dimension classification
    device: str,
    dim_threshold: float = 0.50,
) -> dict:
    """
    Ask VLM a question about an image using a dimension-tailored prompt.

    Returns a trace dict:
        question_en, dimension_id, dimension_confidence,
        full_prompt, vlm_raw_answer_en, used_fallback, fallback_prompt,
        vlm_answer_en, vlm_answer_he
    """
    img = pil_image.copy()
    img.thumbnail((640, 640), Image.Resampling.LANCZOS)

    question_en = re.sub(
        r"\s+", " ",
        translate_text(question_he, he_en_tokenizer, he_en_model, device)
    ).strip()

    concept = _extract_concept(question_en)

    # ── dimension classification ──────────────────────────────────────────────
    dim_id, dim_conf, prompt, enc_prefix = classify_dimension(
        embed_model, concept, threshold=dim_threshold
    )

    is_enc_dec = getattr(vlm.config, "is_encoder_decoder", False)

    inputs = processor(images=img, text=prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    n_input = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = vlm.generate(
            **inputs,
            max_new_tokens=100,
            min_new_tokens=8,
            do_sample=False,
            repetition_penalty=1.5,
        )

    raw_tokens = out[0] if is_enc_dec else out[0][n_input:]
    decoded = _deduplicate_sentences(
        processor.decode(raw_tokens, skip_special_tokens=True).strip()
    )
    vlm_raw_answer_en = (enc_prefix + decoded) if (is_enc_dec and enc_prefix) else decoded
    answer_en = vlm_raw_answer_en
    used_fallback = False
    fallback_prompt_used = ""

    # ── fallback if degenerate ────────────────────────────────────────────────
    if _is_degenerate_answer(answer_en):
        used_fallback = True
        fallback_prompt_used = (
            f'Without using the words "{concept}", describe in one sentence '
            "what concrete objects, actions, or relationships are visible in this image."
        )
        fb_inputs = processor(images=img, text=fallback_prompt_used, return_tensors="pt")
        if device == "cuda":
            fb_inputs = {k: v.to(device) for k, v in fb_inputs.items()}
        n_fb = fb_inputs["input_ids"].shape[1]
        with torch.no_grad():
            fb_out = vlm.generate(
                **fb_inputs,
                max_new_tokens=64, min_new_tokens=8,
                do_sample=False, repetition_penalty=1.5,
            )
        fb_raw = fb_out[0] if is_enc_dec else fb_out[0][n_fb:]
        fb_decoded = _deduplicate_sentences(
            processor.decode(fb_raw, skip_special_tokens=True).strip()
        )
        answer_en = ("This image shows " + fb_decoded) if is_enc_dec else fb_decoded

    if _is_degenerate_answer(answer_en):
        answer_en = (
            "The image shows people interacting, and their interactions "
            "can influence later outcomes."
        )

    answer_he = translate_text(answer_en, en_he_tokenizer, en_he_model, device)

    return {
        "question_en":          question_en,
        "dimension_id":         dim_id,
        "dimension_confidence": round(dim_conf, 4),
        "full_prompt":          prompt,
        "vlm_raw_answer_en":    vlm_raw_answer_en,
        "used_fallback":        used_fallback,
        "fallback_prompt":      fallback_prompt_used,
        "vlm_answer_en":        answer_en,
        "vlm_answer_he":        answer_he,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING  (identical to v1)
# ══════════════════════════════════════════════════════════════════════════════

def compute_similarity(embed_model: SentenceTransformer, text_a: str, text_b: str) -> float:
    a = normalize_hebrew(text_a)
    b = normalize_hebrew(text_b)
    emb = embed_model.encode([a, b], convert_to_tensor=True)
    return util.cos_sim(emb[0], emb[1]).item()


def compute_bertscore(scorer: BERTScorer, hyp: str, ref: str) -> tuple:
    if not hyp.strip() or not ref.strip():
        return 0.0, 0.0, 0.0
    try:
        P, R, F = scorer.score([hyp], [ref])
        return round(P[0].item(), 4), round(R[0].item(), 4), round(F[0].item(), 4)
    except Exception:
        return 0.0, 0.0, 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DATASET LOOP
# ══════════════════════════════════════════════════════════════════════════════

def process_dataset(
    dataset_root: Path,
    vlm_name: str,
    embed_name: str,
    device: str,
    output_path: Path,
    csv_path: Path,
    dim_threshold: float = 0.50,
):
    print(f"\nInitialising dataset from: {dataset_root}")
    dataset = SystemsThinkingDataset(dataset_root)
    print(f"Found {len(dataset)} samples\n")

    # Load all models
    vlm, processor           = load_vlm(vlm_name, device)
    embed_model              = load_embed(embed_name, device)
    labse_model              = load_embed(LABSE_MODEL, device)
    he_en_tokenizer, he_en_mdl = load_translation_model(HE_EN_MODEL, device)
    en_he_tokenizer, en_he_mdl = load_translation_model(EN_HE_MODEL, device)
    print("Loading BERTScorer (roberta-large) ...")
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=False, device="cpu")
    print("[OK] BERTScorer loaded\n")

    # Pre-warm dimension anchor cache using the already-loaded embed_model
    print("Pre-computing dimension anchor embeddings ...")
    from src.prompt_dimensions import _build_cache
    _build_cache(embed_model)
    print("[OK] Anchor cache ready\n")

    results = []
    meta = {"device": device, "vlm": vlm_name, "embed": embed_name,
            "he_en_model": HE_EN_MODEL, "en_he_model": EN_HE_MODEL,
            "dim_threshold": dim_threshold}

    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer   = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    try:
        for sample in tqdm(dataset, total=len(dataset), desc="Samples", unit="sample"):
            q_num       = sample["question_num"]
            a_num       = sample["answer_num"]
            p_num       = sample["participant_num"]
            pil_image   = sample["image"]
            question    = sample["question"]
            subj_answer = sample["answer"]
            image_path  = str(sample["image_path"])
            stem        = f"{q_num}_{a_num}_{p_num}"
            ts          = datetime.utcnow().isoformat(timespec="seconds")

            trace = ask_vlm(
                vlm, processor, pil_image, question,
                he_en_tokenizer, he_en_mdl,
                en_he_tokenizer, en_he_mdl,
                embed_model, device, dim_threshold,
            )

            subj_answer_en = translate_text(subj_answer, he_en_tokenizer, he_en_mdl, device)

            # 4 similarity metrics
            score       = compute_similarity(embed_model, trace["vlm_answer_he"], subj_answer)
            sim_pct     = round(max(0.0, min(1.0, score)) * 100, 2)
            score_labse = compute_similarity(labse_model, trace["vlm_answer_he"], subj_answer)
            sim_labse_pct = round(max(0.0, min(1.0, score_labse)) * 100, 2)
            score_en    = compute_similarity(embed_model, trace["vlm_answer_en"], subj_answer_en)
            sim_en_pct  = round(max(0.0, min(1.0, score_en)) * 100, 2)
            bs_p, bs_r, bs_f1 = compute_bertscore(bert_scorer, trace["vlm_answer_en"], subj_answer_en)

            row = {
                "stem": stem,
                "question_num": q_num,
                "answer_num":   a_num,
                "participant_num": p_num,
                "image_path":   image_path,
                "question_he":  question,
                "question_en":  trace["question_en"],
                "dimension_id": trace["dimension_id"],
                "dimension_confidence": trace["dimension_confidence"],
                "full_prompt":  trace["full_prompt"],
                "vlm_raw_answer_en": trace["vlm_raw_answer_en"],
                "used_fallback":     trace["used_fallback"],
                "fallback_prompt":   trace["fallback_prompt"],
                "vlm_answer_en":     trace["vlm_answer_en"],
                "vlm_answer_he":     trace["vlm_answer_he"],
                "subject_answer_he": subj_answer,
                "similarity":        score,
                "similarity_percent": sim_pct,
                "comparison_method": "multilingual semantic embedding cosine similarity",
                "subject_answer_en": subj_answer_en,
                "similarity_labse_he": score_labse,
                "similarity_labse_he_percent": sim_labse_pct,
                "similarity_en":     score_en,
                "similarity_en_percent": sim_en_pct,
                "bertscore_precision": bs_p,
                "bertscore_recall":    bs_r,
                "bertscore_f1":        bs_f1,
                "vlm_model":    vlm_name,
                "embed_model":  embed_name,
                "he_en_model":  HE_EN_MODEL,
                "en_he_model":  EN_HE_MODEL,
                "device":       device,
                "timestamp":    ts,
            }

            results.append(row)
            writer.writerow(row)
            csv_file.flush()

            tqdm.write(
                f"{stem}  dim={trace['dimension_id']} ({trace['dimension_confidence']:.2f})"
                f"  mpnet={score:.3f}"
            )

            # Incremental JSON
            output_path.write_text(
                json.dumps({**meta, "results": results}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    finally:
        csv_file.close()

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline v2 — dimension-classified concept-tailored prompts"
    )
    parser.add_argument("--dataset",    default="dataset")
    parser.add_argument("--vlm",        default=VLM_MODEL)
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--output",     default=None)
    parser.add_argument("--device",     default=None,
                        help="cpu or cuda (default: auto-detect)")
    parser.add_argument("--dim-threshold", type=float, default=0.50,
                        help="Minimum cosine similarity to accept a dimension match (default: 0.50)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nPipeline v2  |  device: {device}  |  dim_threshold: {args.dim_threshold}\n")

    run_ts      = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else Path(f"results/vlm_results_v2_{run_ts}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path    = output_path.with_suffix(".csv")
    print(f"Output → {output_path}  |  {csv_path}\n")

    results = process_dataset(
        Path(args.dataset), args.vlm, args.embed_model,
        device, output_path, csv_path, args.dim_threshold,
    )
    print(f"\nDone. Processed {len(results)} samples.")
    print(f"  JSON → {output_path}")
    print(f"  CSV  → {csv_path}")


if __name__ == "__main__":
    main()
