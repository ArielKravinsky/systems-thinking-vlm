"""
run_prompt_comparison.py
========================
Run 10 different prompts on every unique (image, question) pair and write a
wide-format CSV + JSON for side-by-side comparison.

Usage (from repo root):
    conda run -n systems_thinking python run_prompt_comparison.py
    conda run -n systems_thinking python run_prompt_comparison.py --device cuda

Output:
    prompt_comparison_YYYYMMDD_HHMMSS.csv   – one row per image-question pair, 10 answer columns
    prompt_comparison_YYYYMMDD_HHMMSS.json  – same data as JSON with prompt texts embedded
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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# ── model constants ────────────────────────────────────────────────────────────
VLM_MODEL   = "Salesforce/blip2-flan-t5-xl"
HE_EN_MODEL = "Helsinki-NLP/opus-mt-tc-big-he-en"

# ── dataset paths ──────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).parent
IMAGES_DIR = REPO_ROOT / "dataset" / "images"
QUES_DIR   = REPO_ROOT / "dataset" / "questions"


# ══════════════════════════════════════════════════════════════════════════════
#  10 PROMPT TEMPLATES
#  Each is a callable: build_prompt(concept: str, question_en: str) -> str
#  The convention for enc-dec (Flan-T5) is:
#    - if a prompt ends with a partial sentence, `prefix` is the text to
#      prepend back to the raw model output so the stored answer is complete.
#    - otherwise prefix = "".
# ══════════════════════════════════════════════════════════════════════════════

PROMPTS = [
    # ── P1: Plain description – no concept hint ────────────────────────────
    {
        "id": "P1",
        "style": "plain_description",
        "description": "Baseline – describe what you see, no concept hint",
        "build": lambda concept, question_en: (
            "Describe what you see in this image in one or two sentences."
        ),
        "prefix": "",
    },
    # ── P2: Single-sentence concept connection ─────────────────────────────
    {
        "id": "P2",
        "style": "concept_connection",
        "description": "Name the concept, ask for a one-sentence connection",
        "build": lambda concept, question_en: (
            f'The concept is: "{concept}".\n'
            "In one sentence, describe what this image shows and how it relates to this concept."
        ),
        "prefix": "",
    },
    # ── P3: Fill-in-blank (current production prompt) ──────────────────────
    {
        "id": "P3",
        "style": "fill_in_blank",
        "description": "Fill-in-the-blank – forces 'shows X because Y' structure (current prompt)",
        "build": lambda concept, question_en: (
            "You are a systems thinking expert analyzing an image.\n"
            f'Concept: "{concept}"\n'
            "Complete this sentence by filling in what you observe and why it connects to the concept:\n"
            f'"This image shows [what you see], which illustrates \\"{concept}\\" because [explain the visual link]."\n'
            "Answer: This image shows"
        ),
        "prefix": "This image shows ",   # re-attach for enc-dec
    },
    # ── P4: Cause-and-effect ─────────────────────────────────────────────
    {
        "id": "P4",
        "style": "cause_effect",
        "description": "Ask explicitly for cause-and-effect reasoning",
        "build": lambda concept, question_en: (
            f'Concept: "{concept}".\n'
            "Look at this image and answer:\n"
            "What cause-and-effect relationship does it show, and how does that illustrate the concept?"
        ),
        "prefix": "",
    },
    # ── P5: Expert narrator (educator mode) ──────────────────────────────
    {
        "id": "P5",
        "style": "expert_narrator",
        "description": "Frame the VLM as an educator explaining to a student",
        "build": lambda concept, question_en: (
            f'You are a systems thinking educator teaching the concept "{concept}".\n'
            "Write one sentence explaining to a student what this image shows and why it was chosen."
        ),
        "prefix": "",
    },
    # ── P6: Direct question (translated question as-is) ──────────────────
    {
        "id": "P6",
        "style": "direct_question",
        "description": "Pass the translated question directly to the model",
        "build": lambda concept, question_en: question_en,
        "prefix": "",
    },
    # ── P7: Why-focused evidence prompt ──────────────────────────────────
    {
        "id": "P7",
        "style": "why_evidence",
        "description": "Ask 'why does this image represent X?' + request visual evidence",
        "build": lambda concept, question_en: (
            f'Why does this image represent the idea that "{concept}"?\n'
            "What specific visual elements support your answer?"
        ),
        "prefix": "",
    },
    # ── P8: Connections / interdependencies ──────────────────────────────
    {
        "id": "P8",
        "style": "connections",
        "description": "Focus on visible connections and interdependencies",
        "build": lambda concept, question_en: (
            f'Concept: "{concept}".\n'
            "What connections, relationships, or interdependencies are visible in this image that illustrate this concept?\n"
            "Answer in one sentence."
        ),
        "prefix": "",
    },
    # ── P9: Narrative / storytelling ─────────────────────────────────────
    {
        "id": "P9",
        "style": "narrative",
        "description": "Ask the model to tell a brief story about the image",
        "build": lambda concept, question_en: (
            f'Tell a brief story (one or two sentences) about what is happening in this image '
            f'that demonstrates the concept "{concept}".'
        ),
        "prefix": "",
    },
    # ── P10: Counterfactual contrast ──────────────────────────────────────
    {
        "id": "P10",
        "style": "counterfactual",
        "description": "Contrast: what IS shown vs. what would change if the concept were absent",
        "build": lambda concept, question_en: (
            "Look at this image and answer in one sentence:\n"
            f'What is happening here, and how would the situation be different if "{concept}" were NOT the case?'
        ),
        "prefix": "",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_vlm(model_name: str, device: str):
    print(f"\nLoading VLM: {model_name} ...")
    if device == "cuda":
        vlm = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
    else:
        vlm = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map="cpu"
        )
    processor = Blip2Processor.from_pretrained(model_name)
    print("[OK] VLM ready")
    return vlm, processor


def load_translation_model(model_name: str, device: str):
    print(f"Loading translation model: {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device == "cuda":
        mdl = mdl.to(device)
    print("[OK] Translation model ready")
    return tok, mdl


def translate_text(text: str, tokenizer, model, device: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    inputs = tokenizer(text, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def _extract_concept(question_en: str) -> str:
    m = re.search(r'["\u201c\u201d\u2018\u2019]([^"\u201c\u201d\u2018\u2019]+)["\u201c\u201d\u2018\u2019]', question_en)
    return m.group(1).strip() if m else question_en


def _deduplicate_sentences(text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    seen_set, unique = set(), []
    for s in sentences:
        key = s.strip().lower()
        if key and key not in seen_set:
            seen_set.add(key)
            unique.append(s.strip())
    return " ".join(unique)


def run_prompts_batched(vlm, processor, pil_image: Image.Image,
                        prompt_defs: list, device: str, batch_size: int = 10) -> list:
    """Run all prompts for one image in (mini-)batched forward passes.

    batch_size controls how many prompts are sent to the GPU at once.
    Returns a list of answer strings in the same order as prompt_defs.
    """
    img = pil_image.copy()
    img.thumbnail((640, 640), Image.Resampling.LANCZOS)
    is_enc_dec = getattr(vlm.config, "is_encoder_decoder", False)

    all_answers = []
    for i in range(0, len(prompt_defs), batch_size):
        chunk = prompt_defs[i: i + batch_size]
        prompt_texts = [p["build"].__call__(p["_concept"], p["_question_en"]) for p in chunk]
        prefixes     = [p["prefix"] for p in chunk]
        images       = [img] * len(prompt_texts)

        inputs = processor(images=images, text=prompt_texts,
                           return_tensors="pt", padding=True, truncation=True)
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        n_input = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = vlm.generate(
                **inputs,
                max_new_tokens=80,
                min_new_tokens=8,
                do_sample=False,
                repetition_penalty=1.5,
            )

        for j, tokens in enumerate(out):
            raw_tokens = tokens if is_enc_dec else tokens[n_input:]
            decoded = _deduplicate_sentences(
                processor.decode(raw_tokens, skip_special_tokens=True).strip()
            )
            prefix = prefixes[j]
            if is_enc_dec and prefix:
                decoded = prefix + decoded
            all_answers.append(decoded)

    return all_answers


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Prompt comparison experiment")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                        help="Device to run inference on (default: cuda)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Prompts per GPU batch (default: 10 = all at once; reduce if OOM)")
    args = parser.parse_args()
    device = args.device
    batch_size = args.batch_size
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        device = "cpu"

    # ── load models ───────────────────────────────────────────────────────────
    gc.collect()
    vlm, processor = load_vlm(VLM_MODEL, device)
    he_en_tok, he_en_mdl = load_translation_model(HE_EN_MODEL, device)

    # ── discover unique (question_num, answer_num) pairs ─────────────────────
    pairs = []
    for img_path in sorted(IMAGES_DIR.glob("*.png")):
        stem = img_path.stem              # e.g. "3_2"
        parts = stem.split("_")
        if len(parts) != 2:
            continue
        q_num, a_num = parts[0], parts[1]
        q_file = QUES_DIR / f"{q_num}.txt"
        if not q_file.exists():
            continue
        question_he = q_file.read_text(encoding="utf-8").strip()
        pairs.append({
            "question_num": int(q_num),
            "answer_num":   int(a_num),
            "image_path":   str(img_path),
            "question_he":  question_he,
        })

    pairs.sort(key=lambda r: (r["question_num"], r["answer_num"]))
    print(f"\nFound {len(pairs)} unique image-question pairs")

    # ── output paths ─────────────────────────────────────────────────────────
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv   = REPO_ROOT / f"prompt_comparison_{ts}.csv"
    out_json  = REPO_ROOT / f"prompt_comparison_{ts}.json"

    # Build CSV column list
    prompt_cols = []
    for p in PROMPTS:
        prompt_cols.append(f"{p['id']}_answer")

    csv_cols = (
        ["question_num", "answer_num", "image_path", "question_he", "question_en", "concept"]
        + prompt_cols
    )

    results = []

    with open(out_csv, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=csv_cols)
        writer.writeheader()

        for pair in tqdm(pairs, desc="Image-question pairs"):
            pil_image    = Image.open(pair["image_path"]).convert("RGB")
            question_he  = pair["question_he"]
            question_en  = translate_text(question_he, he_en_tok, he_en_mdl, device)
            question_en  = re.sub(r"\s+", " ", question_en).strip()
            concept      = _extract_concept(question_en)

            row = {
                "question_num": pair["question_num"],
                "answer_num":   pair["answer_num"],
                "image_path":   pair["image_path"],
                "question_he":  question_he,
                "question_en":  question_en,
                "concept":      concept,
            }
            row_json = dict(row)
            row_json["prompts"] = {}

            print(f"\n  Q{pair['question_num']}_A{pair['answer_num']}  concept: {concept!r}")

            # Stamp concept + question_en into each prompt def so the batch runner can build them
            enriched = []
            for p_def in PROMPTS:
                ep = dict(p_def)
                ep["_concept"]     = concept
                ep["_question_en"] = question_en
                enriched.append(ep)

            # Run all 10 prompts in a single batched forward pass
            answers = run_prompts_batched(vlm, processor, pil_image, enriched, device, batch_size)

            for p_def, answer in zip(PROMPTS, answers):
                col = f"{p_def['id']}_answer"
                row[col] = answer
                prompt_text = p_def["build"](concept, question_en)
                row_json["prompts"][p_def["id"]] = {
                    "style":       p_def["style"],
                    "description": p_def["description"],
                    "prompt_text": prompt_text,
                    "answer":      answer,
                }
                print(f"    {p_def['id']} ({p_def['style']:20s}): {answer[:90]}")

            writer.writerow(row)
            csvf.flush()
            results.append(row_json)

    # ── write JSON ────────────────────────────────────────────────────────────
    json_output = {
        "timestamp":  ts,
        "vlm_model":  VLM_MODEL,
        "he_en_model": HE_EN_MODEL,
        "device":     device,
        "n_pairs":    len(results),
        "prompts_meta": [
            {"id": p["id"], "style": p["style"], "description": p["description"]}
            for p in PROMPTS
        ],
        "results": results,
    }
    out_json.write_text(json.dumps(json_output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Wrote CSV  → {out_csv}")
    print(f"Wrote JSON → {out_json}")
    print(f"Rows: {len(results)}  |  Prompts: {len(PROMPTS)}")


if __name__ == "__main__":
    main()
