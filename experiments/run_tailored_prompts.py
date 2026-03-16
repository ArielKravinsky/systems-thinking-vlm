"""
run_tailored_prompts.py
=======================
For each unique (image, question) pair, run ONE prompt tailored specifically to
that concept's systems-thinking dimension, instead of generic prompts.

Each concept gets its own prompt that explicitly targets:
  - what aspect of systems thinking the concept represents
  - what visual cues are most relevant
  - what kind of explanation is expected

Usage:
    conda run -n systems_thinking python run_tailored_prompts.py
    conda run -n systems_thinking python run_tailored_prompts.py --device cuda

Output:
    tailored_prompts_YYYYMMDD_HHMMSS.csv
    tailored_prompts_YYYYMMDD_HHMMSS.json
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
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Blip2Processor, Blip2ForConditionalGeneration,
)
from PIL import Image
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

REPO_ROOT  = Path(__file__).parent
IMAGES_DIR = REPO_ROOT / "dataset" / "images"
QUES_DIR   = REPO_ROOT / "dataset" / "questions"

VLM_MODEL   = "Salesforce/blip2-flan-t5-xl"
HE_EN_MODEL = "Helsinki-NLP/opus-mt-tc-big-he-en"


# ══════════════════════════════════════════════════════════════════════════════
#  CONCEPT-TAILORED PROMPTS
#
#  Key: the extracted English concept phrase (after translation + _extract_concept)
#  Each entry is a dict:
#    "prompt"      – the full prompt string (receives image via processor)
#    "enc_prefix"  – text to prepend to Flan-T5's output (use "" if not a completion)
#    "rationale"   – why this prompt is specific to this concept (for documentation)
#
#  Match is performed case-insensitively on the concept key substring.
# ══════════════════════════════════════════════════════════════════════════════

TAILORED_PROMPTS = [
    # ── Q1: Work completed successfully ──────────────────────────────────────
    # Focus: cooperation between multiple agents, division of labor, joint completion
    {
        "concept_key": "work completed successfully",
        "question_num": 1,
        "rationale": (
            "This concept is about how multiple agents or components cooperate "
            "to accomplish a shared goal. The prompt asks the model to identify "
            "the actors, their roles, and the coordination that makes completion possible."
        ),
        "prompt": (
            "This image was chosen to illustrate the idea that a task was completed "
            "successfully through cooperation.\n"
            "Answer these in one sentence: Who or what are the actors in this image, "
            "how do they divide their roles, and what does their joint effort achieve?"
        ),
        "enc_prefix": "",
    },

    # ── Q2: The decision has been made ───────────────────────────────────────
    # Focus: collective deliberation, consensus-building, group dynamics in decision-making
    {
        "concept_key": "the decision has been made",
        "question_num": 2,
        "rationale": (
            "Decisions in systems thinking involve feedback, deliberation, and "
            "consensus among multiple stakeholders. The prompt probes the social "
            "process visible in the image rather than just 'people meeting'."
        ),
        "prompt": (
            "This image illustrates the systems thinking idea that a collective "
            "decision has been reached.\n"
            "In one sentence: What social or organizational process is visible here, "
            "and what signals in the image indicate that a shared conclusion was reached?"
        ),
        "enc_prefix": "",
    },

    # ── Q3: The child has learned an important lesson ─────────────────────────
    # Focus: learning as an emergent process, role of environment/interaction, feedback
    {
        "concept_key": "the child has learned an important lesson",
        "question_num": 3,
        "rationale": (
            "Learning is a feedback-driven process shaped by environment, mentors, "
            "and peers. The prompt asks the model to identify the learning dynamic "
            "rather than just 'kids studying'."
        ),
        "prompt": (
            "In systems thinking, learning emerges from interaction with others and "
            "the environment.\n"
            "In one sentence: What interaction or relationship in this image enables "
            "learning, and what clue shows that understanding has been gained?"
        ),
        "enc_prefix": "",
    },

    # ── Q4: The ecosystem is functioning well ──────────────────────────────
    # Focus: interdependence, feedback loops, balance, each part supporting the whole
    {
        "concept_key": "the ecosystem is functioning well",
        "question_num": 4,
        "rationale": (
            "A well-functioning ecosystem depends on interdependent parts and "
            "feedback loops. The prompt asks for the interdependencies visible "
            "in the image, not just a description of what is there."
        ),
        "prompt": (
            "A well-functioning ecosystem relies on the interdependence of its parts.\n"
            "In one sentence: What two or more elements in this image depend on each "
            "other, and how does their interaction keep the system healthy?"
        ),
        "enc_prefix": "",
    },

    # ── Q5: Similar systems ───────────────────────────────────────────────
    # Focus: structural isomorphism — two systems share the same organization/pattern
    {
        "concept_key": "similar systems",
        "question_num": 5,
        "rationale": (
            "'Similar systems' in systems thinking means structural isomorphism: "
            "two different-looking systems share the same underlying organization. "
            "The prompt pushes the model to compare structures, not just appearances."
        ),
        "prompt": (
            "In systems thinking, 'similar systems' means two very different things "
            "share the same underlying structure or pattern of relationships.\n"
            "In one sentence: What two systems or structures appear in this image, "
            "and what organizational pattern do they share despite looking different?"
        ),
        "enc_prefix": "",
    },

    # ── Q6: The group was able to advance towards the goal ────────────────
    # Focus: collective momentum, coordinated effort, movement through shared direction
    {
        "concept_key": "the group was able to advance towards the goal",
        "question_num": 6,
        "rationale": (
            "Collective progress requires synchronized effort and a shared direction. "
            "The prompt targets what makes the group move as a unit, not just "
            "'people walking together'."
        ),
        "prompt": (
            "This image shows collective progress toward a shared goal — a key "
            "concept in systems thinking.\n"
            "In one sentence: What synchronized behavior or shared direction do the "
            "members of the group show, and what obstacle or challenge are they "
            "overcoming together?"
        ),
        "enc_prefix": "",
    },

    # ── Q7: The way I understand reality affects what I see ───────────────
    # Focus: mental models / observer effect — our internal model shapes perception
    {
        "concept_key": "the way i understand reality affects what i see",
        "question_num": 7,
        "rationale": (
            "This is the 'mental models' concept: our internal representation of "
            "the world filters and shapes what we perceive. The prompt asks for "
            "a concrete visual demonstration of that filter or perspective shift."
        ),
        "prompt": (
            "In systems thinking, our mental model of the world shapes what we "
            "notice and how we interpret it.\n"
            "In one sentence: What visual element in this image represents the idea "
            "that the observer's perspective or inner model changes what they perceive?"
        ),
        "enc_prefix": "",
    },

    # ── Q8: When I try to understand what is happening I see things differently
    # Focus: reframing / emergent insight — stepping back reveals a new pattern
    {
        "concept_key": "when i try to understand what is happening",
        "question_num": 8,
        "rationale": (
            "This concept is about the cognitive shift that happens when we step "
            "back and look at a situation from a new angle. The prompt asks for "
            "the dual-interpretation or contrast visible in the image."
        ),
        "prompt": (
            "This image illustrates the idea that trying to understand a situation "
            "can reveal it in a completely new way.\n"
            "In one sentence: How does this image present two different readings or "
            "perspectives of the same scene, and what triggers the shift between them?"
        ),
        "enc_prefix": "",
    },

    # ── Q9: Man affects the world around him ─────────────────────────────
    # Focus: human agency, feedback, unintended consequences, ripple effects
    {
        "concept_key": "man affects the world around him",
        "question_num": 9,
        "rationale": (
            "Human impact in systems thinking includes both intended actions and "
            "feedback loops / unintended consequences. The prompt asks for the "
            "causal chain: what the person does and what changes in the world."
        ),
        "prompt": (
            "In systems thinking, every human action ripples through the system "
            "and creates change beyond its immediate target.\n"
            "In one sentence: What action does the person or people in this image "
            "take, and what broader effect on the surrounding world does it cause "
            "or symbolize?"
        ),
        "enc_prefix": "",
    },

    # ── Q10: Change is made ────────────────────────────────────────────────
    # Focus: process of transformation over time — stages, contrast before/after, gradual shift
    {
        "concept_key": "change is made",
        "question_num": 10,
        "rationale": (
            "'Change is made' in systems thinking is about transformation as a "
            "process rather than an event: accumulation, stages, before-and-after. "
            "The prompt asks the model to identify the transformation dynamic."
        ),
        "prompt": (
            "In systems thinking, change is a gradual process: stages accumulate "
            "until a new state emerges.\n"
            "In one sentence: What transformation or contrast between states is "
            "visible in this image, and what does it suggest about how change happens "
            "over time?"
        ),
        "enc_prefix": "",
    },
]

# Build lookup: question_num → tailored prompt definition
PROMPT_BY_QNUM = {entry["question_num"]: entry for entry in TAILORED_PROMPTS}


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


def run_inference(vlm, processor, pil_image: Image.Image,
                  prompt_text: str, enc_prefix: str, device: str) -> str:
    img = pil_image.copy()
    img.thumbnail((640, 640), Image.Resampling.LANCZOS)
    is_enc_dec = getattr(vlm.config, "is_encoder_decoder", False)

    inputs = processor(images=img, text=prompt_text, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    n_input = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = vlm.generate(
            **inputs,
            max_new_tokens=100,
            min_new_tokens=10,
            do_sample=False,
            repetition_penalty=1.5,
        )

    raw_tokens = out[0] if is_enc_dec else out[0][n_input:]
    decoded = _deduplicate_sentences(
        processor.decode(raw_tokens, skip_special_tokens=True).strip()
    )
    if is_enc_dec and enc_prefix:
        return enc_prefix + decoded
    return decoded


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Concept-tailored prompt experiment")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        device = "cpu"

    gc.collect()
    vlm, processor    = load_vlm(VLM_MODEL, device)
    he_en_tok, he_en_mdl = load_translation_model(HE_EN_MODEL, device)

    # ── discover pairs ────────────────────────────────────────────────────────
    pairs = []
    for img_path in sorted(IMAGES_DIR.glob("*.png")):
        stem  = img_path.stem
        parts = stem.split("_")
        if len(parts) != 2:
            continue
        q_num, a_num = int(parts[0]), int(parts[1])
        q_file = QUES_DIR / f"{q_num}.txt"
        if not q_file.exists():
            continue
        if q_num not in PROMPT_BY_QNUM:
            print(f"[SKIP] No tailored prompt defined for Q{q_num}")
            continue
        question_he = q_file.read_text(encoding="utf-8").strip()
        pairs.append(dict(question_num=q_num, answer_num=a_num,
                          image_path=str(img_path), question_he=question_he))

    pairs.sort(key=lambda r: (r["question_num"], r["answer_num"]))
    print(f"\nFound {len(pairs)} image-question pairs across {len(PROMPT_BY_QNUM)} concepts")

    # ── output ────────────────────────────────────────────────────────────────
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv  = REPO_ROOT / f"tailored_prompts_{ts}.csv"
    out_json = REPO_ROOT / f"tailored_prompts_{ts}.json"

    CSV_COLS = [
        "question_num", "answer_num", "image_path",
        "question_he", "question_en", "concept",
        "prompt_rationale", "prompt_text", "vlm_answer",
    ]

    results = []

    with open(out_csv, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=CSV_COLS)
        writer.writeheader()

        for pair in tqdm(pairs, desc="Pairs"):
            q_num      = pair["question_num"]
            pil_image  = Image.open(pair["image_path"]).convert("RGB")
            question_en = re.sub(r"\s+", " ",
                translate_text(pair["question_he"], he_en_tok, he_en_mdl, device)
            ).strip()
            concept    = _extract_concept(question_en)

            p_def      = PROMPT_BY_QNUM[q_num]
            prompt_txt = p_def["prompt"]
            enc_prefix = p_def["enc_prefix"]

            answer = run_inference(vlm, processor, pil_image, prompt_txt, enc_prefix, device)

            print(f"\n  Q{q_num}_A{pair['answer_num']}  [{concept}]")
            print(f"  → {answer}")

            row = dict(
                question_num   = q_num,
                answer_num     = pair["answer_num"],
                image_path     = pair["image_path"],
                question_he    = pair["question_he"],
                question_en    = question_en,
                concept        = concept,
                prompt_rationale = p_def["rationale"],
                prompt_text    = prompt_txt,
                vlm_answer     = answer,
            )
            writer.writerow(row)
            csvf.flush()
            results.append(row)

    # ── JSON ──────────────────────────────────────────────────────────────────
    json_out = {
        "timestamp":   ts,
        "vlm_model":   VLM_MODEL,
        "he_en_model": HE_EN_MODEL,
        "device":      device,
        "n_pairs":     len(results),
        "prompts_by_question": {
            str(e["question_num"]): {
                "concept_key": e["concept_key"],
                "rationale":   e["rationale"],
                "prompt":      e["prompt"],
            }
            for e in TAILORED_PROMPTS
        },
        "results": results,
    }
    out_json.write_text(json.dumps(json_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Wrote CSV  → {out_csv}")
    print(f"Wrote JSON → {out_json}")
    print(f"Pairs: {len(results)}")


if __name__ == "__main__":
    main()
