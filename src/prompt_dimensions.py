"""
src/prompt_dimensions.py
========================
Dimension classifier for systems-thinking concepts.

Every systems-thinking question belongs to one of 8 structural dimensions.
For each dimension we define:
  - anchor_phrases: representative concepts that exemplify the dimension.
    These are embedded at import time and used for nearest-neighbor lookup.
  - build_prompt(concept): returns (prompt_text, enc_prefix)
    enc_prefix is text to prepend back to Flan-T5 output (empty string = no prefix).

classify_dimension(embed_model, concept_en, threshold=0.55)
  → (dimension_id, confidence, prompt_text, enc_prefix)

If the best cosine similarity is below `threshold` the function returns
  ("fallback", confidence, <generic fill-in-blank prompt>, "This image shows ")
so the pipeline degrades gracefully to the previous production prompt.
"""

from __future__ import annotations
from typing import Callable
import torch
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------------
# 8 DIMENSION DEFINITIONS
# ---------------------------------------------------------------------------

def _d(id_: str, anchors: list[str], build: Callable[[str], tuple[str, str]]) -> dict:
    return {"id": id_, "anchors": anchors, "build": build}


DIMENSIONS: list[dict] = [

    # ── D1: Collective Completion ──────────────────────────────────────────
    # Questions about multiple agents cooperating to finish a shared task.
    _d(
        "collective_completion",
        anchors=[
            "work completed successfully",
            "task accomplished together",
            "job finished through cooperation",
            "mission completed",
            "work was done",
            "the work was completed",
        ],
        build=lambda concept: (
            "This image was chosen to illustrate the idea that a task was completed "
            "successfully through cooperation.\n"
            "Answer in one sentence: Who or what are the actors in this image, "
            "how do they divide their roles, and what does their joint effort achieve?",
            "",
        ),
    ),

    # ── D2: Collective Deliberation ────────────────────────────────────────
    # Questions about group decision-making, consensus, and shared conclusions.
    _d(
        "collective_deliberation",
        anchors=[
            "the decision has been made",
            "a decision was reached",
            "consensus was achieved",
            "agreement was made",
            "they decided together",
            "collective choice",
        ],
        build=lambda concept: (
            "This image illustrates the systems thinking idea that a collective "
            "decision has been reached.\n"
            "In one sentence: What social or organizational process is visible here, "
            "and what signals in the image indicate that a shared conclusion was reached?",
            "",
        ),
    ),

    # ── D3: Learning Through Interaction ──────────────────────────────────
    # Questions about knowledge acquisition shaped by environment, peers, mentors.
    _d(
        "learning_interaction",
        anchors=[
            "the child has learned an important lesson",
            "learning happened through experience",
            "knowledge was gained",
            "the student understood",
            "insight was reached",
            "understanding emerged",
        ],
        build=lambda concept: (
            "In systems thinking, learning emerges from interaction with others and "
            "the environment.\n"
            "In one sentence: What interaction or relationship in this image enables "
            "learning, and what clue shows that understanding has been gained?",
            "",
        ),
    ),

    # ── D4: Ecosystem Interdependence ─────────────────────────────────────
    # Questions about parts of a system depending on each other to stay in balance.
    _d(
        "ecosystem_interdependence",
        anchors=[
            "the ecosystem is functioning well",
            "the system is in balance",
            "parts depend on each other",
            "interdependence keeps the system healthy",
            "mutually beneficial relationship",
            "the environment is thriving",
        ],
        build=lambda concept: (
            "A well-functioning ecosystem relies on the interdependence of its parts.\n"
            "In one sentence: What two or more elements in this image depend on each "
            "other, and how does their interaction keep the system healthy?",
            "",
        ),
    ),

    # ── D5: Structural Isomorphism ────────────────────────────────────────
    # Questions about two different-looking systems sharing the same underlying pattern.
    _d(
        "structural_isomorphism",
        anchors=[
            "similar systems",
            "two systems share the same structure",
            "structural analogy",
            "same pattern different context",
            "isomorphic structures",
            "parallel systems",
            "analogous organization",
        ],
        build=lambda concept: (
            "In systems thinking, 'similar systems' means two very different things "
            "share the same underlying structure or pattern of relationships.\n"
            "In one sentence: What two systems or structures appear in this image, "
            "and what organizational pattern do they share despite looking different?",
            "",
        ),
    ),

    # ── D6: Collective Momentum ────────────────────────────────────────────
    # Questions about a group advancing toward a shared goal through coordinated effort.
    _d(
        "collective_momentum",
        anchors=[
            "the group was able to advance towards the goal",
            "team made progress together",
            "group moved forward as one",
            "collective progress toward target",
            "they advanced together",
            "moving toward a shared objective",
        ],
        build=lambda concept: (
            "This image shows collective progress toward a shared goal — a key "
            "concept in systems thinking.\n"
            "In one sentence: What synchronized behavior or shared direction do the "
            "members of the group show, and what challenge are they overcoming together?",
            "",
        ),
    ),

    # ── D7: Mental Models / Observer Effect ──────────────────────────────
    # Questions about how an observer's inner model shapes what they perceive.
    _d(
        "mental_models",
        anchors=[
            "the way I understand reality affects what I see",
            "my perspective shapes my perception",
            "mental model filters reality",
            "how I see the world changes what I notice",
            "worldview influences observation",
            "perception is shaped by belief",
            "when I try to understand what is happening sometimes I see things differently",
            "trying to understand reveals a new perspective",
            "reframing changes what you see",
            "stepping back reveals a different picture",
        ],
        build=lambda concept: (
            f'In systems thinking, our mental model of the world shapes what we '
            f'notice and how we interpret it. The concept here is: "{concept}".\n'
            "In one sentence: What visual element in this image represents the idea "
            "that an observer's perspective or inner model changes what they perceive, "
            "or that the same scene can be understood in two different ways?",
            "",
        ),
    ),

    # ── D8: Human Agency & Ripple Effects ────────────────────────────────
    # Questions about how human actions propagate through a system.
    _d(
        "human_agency",
        anchors=[
            "man affects the world around him",
            "human action changes the environment",
            "people influence their surroundings",
            "individual impact on the system",
            "actions have consequences beyond their target",
            "humans shape the world",
        ],
        build=lambda concept: (
            "In systems thinking, every human action ripples through the system "
            "and creates change beyond its immediate target.\n"
            "In one sentence: What action does the person or people in this image "
            "take, and what broader effect on the surrounding world does it cause "
            "or symbolize?",
            "",
        ),
    ),

    # ── D9: Transformation Over Time ──────────────────────────────────────
    # Questions about gradual change, stages, before-and-after contrasts.
    _d(
        "transformation",
        anchors=[
            "change is made",
            "transformation occurs",
            "things change over time",
            "stages of change",
            "before and after",
            "gradual shift",
            "evolution of a system",
            "the change was made",
        ],
        build=lambda concept: (
            "In systems thinking, change is a gradual process: stages accumulate "
            "until a new state emerges.\n"
            "In one sentence: What transformation or contrast between states is "
            "visible in this image, and what does it suggest about how change happens "
            "over time?",
            "",
        ),
    ),
]

# ---------------------------------------------------------------------------
# Generic fallback (current production prompt)
# ---------------------------------------------------------------------------

def _fallback_prompt(concept: str) -> tuple[str, str]:
    return (
        "You are a systems thinking expert analyzing an image.\n"
        f'Concept: "{concept}"\n'
        "Complete this sentence by filling in what you observe and why it connects to the concept:\n"
        f'"This image shows [what you see], which illustrates \\"{concept}\\" because [explain the visual link]."\n'
        "Answer: This image shows",
        "This image shows ",
    )


# ---------------------------------------------------------------------------
# CLASSIFIER
# ---------------------------------------------------------------------------

# Anchor embeddings are computed lazily and cached here.
_anchor_cache: dict[str, "torch.Tensor"] = {}   # model_name → stacked tensor
_anchor_meta: list[tuple[int, dict]] = []        # (dim_index, dim_dict) per anchor


def _build_cache(embed_model: SentenceTransformer) -> tuple["torch.Tensor", list[tuple[int, dict]]]:
    """Build (or return cached) stacked anchor embedding tensor."""
    key = embed_model.device if hasattr(embed_model, "device") else "cpu"
    key = str(key)
    if key not in _anchor_cache:
        phrases: list[str] = []
        meta: list[tuple[int, dict]] = []
        for i, dim in enumerate(DIMENSIONS):
            for phrase in dim["anchors"]:
                phrases.append(phrase)
                meta.append((i, dim))
        embs = embed_model.encode(phrases, convert_to_tensor=True, show_progress_bar=False)
        _anchor_cache[key] = embs
        # Store meta alongside (same key prefix)
        _anchor_cache[key + "_meta"] = meta
    return _anchor_cache[key], _anchor_cache[key + "_meta"]


def classify_dimension(
    embed_model: SentenceTransformer,
    concept_en: str,
    threshold: float = 0.50,
) -> tuple[str, float, str, str]:
    """
    Match concept_en to the best systems-thinking dimension.

    Returns:
        (dimension_id, confidence, prompt_text, enc_prefix)

    If best confidence < threshold, falls back to the generic fill-in-blank prompt
    with dimension_id = "fallback".
    """
    concept_clean = concept_en.strip().lower()
    anchor_embs, meta = _build_cache(embed_model)

    concept_emb = embed_model.encode(concept_clean, convert_to_tensor=True, show_progress_bar=False)
    sims = util.cos_sim(concept_emb, anchor_embs)[0]  # shape (n_anchors,)

    best_idx = int(sims.argmax())
    best_score = float(sims[best_idx])
    dim_idx, dim_dict = meta[best_idx]

    if best_score < threshold:
        prompt_text, enc_prefix = _fallback_prompt(concept_en)
        return "fallback", best_score, prompt_text, enc_prefix

    prompt_text, enc_prefix = dim_dict["build"](concept_en)
    return dim_dict["id"], best_score, prompt_text, enc_prefix
