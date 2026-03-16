"""
pipeline_qwen.py
================
VLM pipeline using Qwen2-VL-7B-Instruct.

Key design choices:
  - Question pre-translated He→En before reaching the VLM, so the model focuses
    on visual analysis rather than language translation.
  - Concise prompt: model instructed to answer in ≤3 sentences without preamble.
  - VLM answer cache keyed by (image_path, question_he): same image+question always
    yields the same answer, so only ~40 unique VLM inferences are needed for 539
    samples (~13x speedup vs. per-sample approach).
  - Resume support: --resume <prior_json> skips already-processed stems and
    pre-populates the VLM cache from the prior run, so only new samples are computed.
  - Extra metrics beyond cosine similarity:
      concept_sim_en            – cosine similarity between the SUBJECT answer and the bare
                                  concept phrase (high = subject just echoed the concept word)
      sim_above_concept_scalar  – similarity_en minus concept_sim_en; scalar difference of two
                                  independent cosine projections (approximate baseline adjustment)
      sim_above_concept_vec     – cos(subj − proj_concept(subj), vlm); projects the concept
                                  direction out of the subject embedding before measuring
                                  alignment with the VLM description; geometrically distinct
                                  from the scalar version and NOT equivalent to it
      sim_above_concept_vec_both – cos(subj_perp_concept, vlm_perp_concept); removes concept
                                  direction from BOTH subject and VLM embeddings before
                                  measuring alignment (stricter de-concepted grounding score)
      subject_word_count – word count of Hebrew subject answer (low → likely vague)
      vlm_cached         – True if this sample reused a cached VLM answer
"""

import argparse
import csv
import json
import os
import re
import gc
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from tqdm import tqdm
from bert_score import BERTScorer

from .utils_hebrew import normalize_hebrew
from .dataset_loader import SystemsThinkingDataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# ── model IDs ────────────────────────────────────────────────────────────────
QWEN_MODEL  = "Qwen/Qwen2-VL-7B-Instruct"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LABSE_MODEL = "sentence-transformers/LaBSE"
EN_HE_MODEL = "Helsinki-NLP/opus-mt-en-he"
HE_EN_MODEL = "Helsinki-NLP/opus-mt-tc-big-he-en"

CSV_COLUMNS = [
    "stem", "question_num", "answer_num", "participant_num", "image_path",
    "question_he", "question_en", "concept_text",
    "full_prompt",
    "vlm_raw_answer_en", "used_fallback", "fallback_prompt", "vlm_cached",
    "vlm_answer_en", "vlm_answer_he",
    "subject_answer_he", "subject_word_count",
    "similarity", "similarity_percent", "comparison_method",
    "subject_answer_en",
    "similarity_labse_he", "similarity_labse_he_percent",
    "similarity_en", "similarity_en_percent",
    "concept_sim_en", "sim_above_concept_scalar", "sim_above_concept_vec", "sim_above_concept_vec_both",
    "bertscore_precision", "bertscore_recall", "bertscore_f1",
    "vlm_model", "embed_model", "he_en_model", "en_he_model", "device", "timestamp",
]


# ── loaders ──────────────────────────────────────────────────────────────────

def load_vlm(device: str):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print(f"Loading VLM: {QWEN_MODEL}  (4-bit quantised) ...")
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    vlm = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    print("[OK] Qwen2-VL loaded!\n")
    return vlm, processor


def load_embed(model_name: str, device: str):
    print(f"Loading embedding model: {model_name} ...")
    m = SentenceTransformer(model_name, device=device)
    print("[OK] Embedding model loaded!")
    return m


def load_translation_model(model_name: str, device: str):
    print(f"Loading translation model: {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device == "cuda":
        mdl = mdl.to(device)
    print("[OK] Translation model loaded!")
    return tok, mdl


# ── helpers ──────────────────────────────────────────────────────────────────

def translate_text(text: str, tokenizer, model, device: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def _extract_concept(question_en: str) -> str:
    """Extract the quoted concept from 'Explain why ... reflecting "X"' questions."""
    m = re.search(
        r'["\u201c\u201d\u2018\u2019]([^"\u201c\u201d\u2018\u2019]+)["\u201c\u201d\u2018\u2019]',
        question_en,
    )
    return m.group(1).strip() if m else question_en


def _deduplicate_sentences(text: str) -> str:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    seen, unique = set(), []
    for s in parts:
        key = s.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(s.strip())
    return " ".join(unique)


def _is_degenerate(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t or len(t) < 5:
        return True
    if "\ufffd" in t:
        return True
    words = re.findall(r"[a-zA-Z']+", t)
    if len(words) >= 4 and len(set(words)) / len(words) < 0.35:
        return True
    return False


def compute_similarity(embed_model, a: str, b: str) -> float:
    a = normalize_hebrew(a)
    b = normalize_hebrew(b)
    emb = embed_model.encode([a, b], convert_to_tensor=True)
    return util.cos_sim(emb[0], emb[1]).item()


def compute_sac_vec(embed_model, subj_en: str, vlm_en: str, concept: str) -> float:
    """Vector-projection SAC: cos( subj − proj_{concept}(subj),  vlm ).

    Unlike sim_above_concept_scalar = cos(subj,vlm) − cos(subj,concept), which
    subtracts two independent scalar projections, this removes the concept
    direction from the subject embedding *before* measuring alignment with
    the VLM description.  The two metrics capture different geometry and are
    NOT equivalent: infinitely many (subj, vlm, concept) triples share the same
    scalar SAC while producing different sac_vec values.

    Returns 0.0 when the subject embedding is entirely parallel to the concept
    (i.e. the subject answer contains nothing beyond the concept label).
    """
    import torch.nn.functional as F
    texts = [normalize_hebrew(subj_en), normalize_hebrew(vlm_en), normalize_hebrew(concept)]
    embs = embed_model.encode(texts, convert_to_tensor=True)
    e_subj    = F.normalize(embs[0], dim=0)
    e_vlm     = F.normalize(embs[1], dim=0)
    e_concept = F.normalize(embs[2], dim=0)
    # Remove concept component from subject
    proj   = torch.dot(e_subj, e_concept)
    e_perp = e_subj - proj * e_concept
    norm   = torch.norm(e_perp)
    if norm < 1e-8:
        return 0.0
    e_perp = e_perp / norm
    return round(torch.dot(e_perp, e_vlm).item(), 4)


def compute_sac_vec_both(embed_model, subj_en: str, vlm_en: str, concept: str) -> float:
    """Bi-projected SAC: cos( subj_perp_concept, vlm_perp_concept ).

    Removes the concept direction from BOTH subject and VLM embeddings before
    comparison. This yields a stricter measure of alignment after de-concepting
    both sides.

    Returns 0.0 if either de-concepted vector is near-zero norm.
    """
    import torch.nn.functional as F
    texts = [normalize_hebrew(subj_en), normalize_hebrew(vlm_en), normalize_hebrew(concept)]
    embs = embed_model.encode(texts, convert_to_tensor=True)
    e_subj    = F.normalize(embs[0], dim=0)
    e_vlm     = F.normalize(embs[1], dim=0)
    e_concept = F.normalize(embs[2], dim=0)

    proj_subj = torch.dot(e_subj, e_concept)
    proj_vlm  = torch.dot(e_vlm, e_concept)

    subj_perp = e_subj - proj_subj * e_concept
    vlm_perp  = e_vlm  - proj_vlm  * e_concept

    subj_norm = torch.norm(subj_perp)
    vlm_norm  = torch.norm(vlm_perp)
    if subj_norm < 1e-8 or vlm_norm < 1e-8:
        return 0.0

    subj_perp = subj_perp / subj_norm
    vlm_perp  = vlm_perp  / vlm_norm
    return round(torch.dot(subj_perp, vlm_perp).item(), 4)


def compute_bertscore(scorer, hyp: str, ref: str):
    if not hyp.strip() or not ref.strip():
        return 0.0, 0.0, 0.0
    try:
        P, R, F = scorer.score([hyp], [ref])
        return round(P[0].item(), 4), round(R[0].item(), 4), round(F[0].item(), 4)
    except Exception:
        return 0.0, 0.0, 0.0


# ── inference ────────────────────────────────────────────────────────────────

def ask_vlm(
    vlm, processor,
    pil_image: Image.Image,
    question_en: str,
    concept: str,
    en_he_tokenizer, en_he_model,
    device: str,
) -> dict:
    """Run Qwen2-VL on one (image, question) pair.
    question_en and concept are pre-computed and passed in by the caller.
    Returns a trace dict (no question_en/concept_text — caller adds those).
    """
    from qwen_vl_utils import process_vision_info

    img = pil_image.copy()
    img.thumbnail((640, 640), Image.Resampling.LANCZOS)

    # Concise, direct prompt — question already in English, no translation preamble
    prompt_text = (
        "You are a systems thinking expert. "
        "Look at the image and answer the question in 2-3 concise sentences. "
        "Describe specific visual elements you observe and explain how they connect "
        "to the concept. Do not restate or translate the question.\n\n"
        f'Concept: "{concept}"\n'
        f"Question: {question_en}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": prompt_text},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if device == "cuda":
        inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = vlm.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            repetition_penalty=1.3,
        )

    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    raw_answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    answer_en = _deduplicate_sentences(raw_answer)
    used_fallback = False
    fallback_prompt_used = ""

    if _is_degenerate(answer_en):
        used_fallback = True
        fallback_text = (
            "Describe what you see in this image: what objects, people, or "
            "processes are shown, and how they relate to each other?"
        )
        fb_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": fallback_text},
                ],
            }
        ]
        fb_text_input = processor.apply_chat_template(
            fb_messages, tokenize=False, add_generation_prompt=True
        )
        fb_image_inputs, fb_video_inputs = process_vision_info(fb_messages)
        fb_inputs = processor(
            text=[fb_text_input],
            images=fb_image_inputs,
            videos=fb_video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if device == "cuda":
            fb_inputs = fb_inputs.to(device)
        with torch.no_grad():
            fb_ids = vlm.generate(
                **fb_inputs,
                max_new_tokens=200,
                do_sample=False,
                repetition_penalty=1.3,
            )
        fb_trimmed = [out[len(inp):] for inp, out in zip(fb_inputs.input_ids, fb_ids)]
        answer_en = _deduplicate_sentences(
            processor.batch_decode(
                fb_trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        )
        fallback_prompt_used = fallback_text

    if _is_degenerate(answer_en):
        answer_en = "The image depicts a system with interacting components that influence each other's behaviour."

    answer_he = translate_text(answer_en, en_he_tokenizer, en_he_model, device)

    return {
        "full_prompt": prompt_text,
        "vlm_raw_answer_en": raw_answer,
        "used_fallback": used_fallback,
        "fallback_prompt": fallback_prompt_used,
        "vlm_answer_en": answer_en,
        "vlm_answer_he": answer_he,
    }


# ── main processing loop ─────────────────────────────────────────────────────

def process_dataset(dataset_root: Path, embed_name: str, device: str,
                    output_path: Path, csv_path: Path,
                    resume_path: Path = None):
    print(f"\nInitialising dataset from: {dataset_root}")
    dataset = SystemsThinkingDataset(dataset_root)
    print(f"Found {len(dataset)} samples\n")

    vlm, processor           = load_vlm(device)
    embed_model              = load_embed(embed_name, device)
    labse_model              = load_embed(LABSE_MODEL, device)
    he_en_tokenizer, he_en_m = load_translation_model(HE_EN_MODEL, device)
    en_he_tokenizer, en_he_m = load_translation_model(EN_HE_MODEL, device)

    print("Loading BERTScorer ...")
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=False, device="cpu")
    print("[OK] BERTScorer loaded!\n")

    payload_meta = {
        "device": device, "vlm": QWEN_MODEL, "embed": embed_name,
        "he_en_model": HE_EN_MODEL, "en_he_model": EN_HE_MODEL,
    }

    # ── Resume: load previously completed results ─────────────────────────
    results: list = []
    done_stems: set = set()
    vlm_cache: dict = {}

    if resume_path is not None and resume_path.exists():
        prior = json.loads(resume_path.read_text(encoding="utf-8"))
        prior_results = prior.get("results", [])
        for r in prior_results:
            results.append(r)
            done_stems.add(r["stem"])
            # Re-populate VLM cache so new images in the same question skip inference
            cache_key = (r["image_path"], r["question_he"])
            if cache_key not in vlm_cache:
                vlm_cache[cache_key] = {
                    "full_prompt":      r.get("full_prompt", ""),
                    "vlm_raw_answer_en":r.get("vlm_raw_answer_en", ""),
                    "used_fallback":    r.get("used_fallback", False),
                    "fallback_prompt":  r.get("fallback_prompt", ""),
                    "vlm_answer_en":    r.get("vlm_answer_en", ""),
                    "vlm_answer_he":    r.get("vlm_answer_he", ""),
                    "question_en":      r.get("question_en", ""),
                    "concept_text":     r.get("concept_text", ""),
                }
        print(f"Resuming from {resume_path.name}: "
              f"{len(done_stems)} stems already done, "
              f"{len(vlm_cache)} VLM cache entries pre-loaded.\n")

    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    # Write already-completed rows so the CSV stays complete
    for r in results:
        writer.writerow({k: r.get(k, "") for k in CSV_COLUMNS})
    csv_file.flush()

    unique_pairs = len({(str(s["image_path"]), s["question"]) for s in dataset})
    print(f"Processing {len(dataset)} samples "
          f"({unique_pairs} unique image+question pairs → ~{unique_pairs} VLM calls)\n")

    try:
        for sample in tqdm(dataset, total=len(dataset), desc="Qwen2-VL", unit="sample"):
            q_num    = sample["question_num"]
            a_num    = sample["answer_num"]
            p_num    = sample["participant_num"]
            img      = sample["image"]
            q_he     = sample["question"]
            subj_he  = sample["answer"]
            img_path = str(sample["image_path"])
            stem     = f"{q_num}_{a_num}_{p_num}"
            ts       = datetime.utcnow().isoformat(timespec="seconds")

            # ── Skip already-completed stems (resume mode) ─────────────────
            if stem in done_stems:
                continue

            # ── VLM inference (cached per unique image+question) ───────────
            cache_key = (img_path, q_he)
            cached = cache_key in vlm_cache

            if cached:
                trace       = vlm_cache[cache_key]
                question_en = trace["question_en"]
                concept     = trace["concept_text"]
            else:
                question_en = translate_text(q_he, he_en_tokenizer, he_en_m, device)
                question_en = re.sub(r"\s+", " ", question_en).strip()
                concept     = _extract_concept(question_en)
                trace_raw   = ask_vlm(
                    vlm, processor, img, question_en, concept,
                    en_he_tokenizer, en_he_m, device,
                )
                trace = {**trace_raw, "question_en": question_en, "concept_text": concept}
                vlm_cache[cache_key] = trace

            # ── translate subject answer to English ────────────────────────
            subj_en            = translate_text(subj_he, he_en_tokenizer, he_en_m, device)
            subject_word_count = len(subj_he.split())

            # ── similarity metrics ─────────────────────────────────────────
            scr        = compute_similarity(embed_model,  trace["vlm_answer_he"], subj_he)
            scr_labse  = compute_similarity(labse_model,  trace["vlm_answer_he"], subj_he)
            scr_en     = compute_similarity(embed_model,  trace["vlm_answer_en"], subj_en)

            # concept baseline: how similar is the subject answer to the bare concept?
            # High → subject answer just echoed the concept without visual content
            concept_sim_en           = compute_similarity(embed_model, subj_en, concept)
            sim_above_concept_scalar = round(scr_en - concept_sim_en, 4)
            sim_above_concept_vec    = compute_sac_vec(embed_model, subj_en, trace["vlm_answer_en"], concept)
            sim_above_concept_vec_both = compute_sac_vec_both(embed_model, subj_en, trace["vlm_answer_en"], concept)

            bs_p, bs_r, bs_f1 = compute_bertscore(
                bert_scorer, trace["vlm_answer_en"], subj_en
            )

            row = {
                "stem": stem, "question_num": q_num, "answer_num": a_num,
                "participant_num": p_num, "image_path": img_path,
                "question_he": q_he,
                "question_en": question_en,
                "concept_text": concept,
                "full_prompt": trace["full_prompt"],
                "vlm_raw_answer_en": trace["vlm_raw_answer_en"],
                "used_fallback": trace["used_fallback"],
                "fallback_prompt": trace["fallback_prompt"],
                "vlm_cached": cached,
                "vlm_answer_en": trace["vlm_answer_en"],
                "vlm_answer_he": trace["vlm_answer_he"],
                "subject_answer_he": subj_he,
                "subject_word_count": subject_word_count,
                "similarity": scr,
                "similarity_percent": round(max(0.0, min(1.0, scr)) * 100, 2),
                "comparison_method": "multilingual semantic embedding cosine similarity",
                "subject_answer_en": subj_en,
                "similarity_labse_he": scr_labse,
                "similarity_labse_he_percent": round(max(0.0, min(1.0, scr_labse)) * 100, 2),
                "similarity_en": scr_en,
                "similarity_en_percent": round(max(0.0, min(1.0, scr_en)) * 100, 2),
                "concept_sim_en": round(concept_sim_en, 4),
                "sim_above_concept_scalar": sim_above_concept_scalar,
                "sim_above_concept_vec": sim_above_concept_vec,
                "sim_above_concept_vec_both": sim_above_concept_vec_both,
                "bertscore_precision": bs_p,
                "bertscore_recall": bs_r,
                "bertscore_f1": bs_f1,
                "vlm_model": QWEN_MODEL,
                "embed_model": embed_name,
                "he_en_model": HE_EN_MODEL,
                "en_he_model": EN_HE_MODEL,
                "device": device,
                "timestamp": ts,
            }
            results.append(row)
            writer.writerow(row)
            csv_file.flush()

            cache_marker = " [cached]" if cached else ""
            tqdm.write(
                f"{stem}: sim_en={scr_en:.3f}  Δscalar={sim_above_concept_scalar:+.3f}  Δvec={sim_above_concept_vec:+.3f}  Δvec2={sim_above_concept_vec_both:+.3f}"
                f"  subj_words={subject_word_count}{cache_marker}"
            )

            output_path.write_text(
                json.dumps({**payload_meta, "results": results},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    finally:
        csv_file.close()

    # Always write final JSON (covers pure-resume runs where loop body was skipped)
    output_path.write_text(
        json.dumps({**payload_meta, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL-7B-Instruct pipeline")
    parser.add_argument("--dataset",     default="dataset")
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--output",      default=None)
    parser.add_argument("--resume",      default=None,
                        help="Path to a prior results JSON to resume from "
                             "(already-done stems are skipped; VLM cache pre-loaded).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    run_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output \
                  else Path(f"results/vlm_results_qwen_{run_ts}.json")
    csv_path = output_path.with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resume_path = Path(args.resume) if args.resume else None

    print(f"Output → {output_path}  |  {csv_path}\n")
    results = process_dataset(Path(args.dataset), args.embed_model, device,
                              output_path, csv_path, resume_path=resume_path)
    print(f"\nDone – {len(results)} samples → {output_path}")


if __name__ == "__main__":
    main()
