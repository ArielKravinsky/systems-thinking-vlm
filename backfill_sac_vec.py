#!/usr/bin/env python3
"""
backfill_sac_vec.py
===================
Post-process an existing results JSON to:

  1. Rename  sim_above_concept  →  sim_above_concept_scalar
     (scalar difference of two cosine projections — the original metric)

  2. Add     sim_above_concept_vec
     (vector-projection metric: cos(subj − proj_{concept}(subj), vlm))

  3. Add     sim_above_concept_vec_both
      (bi-projected metric: cos(subj_perp_concept, vlm_perp_concept))

Does NOT re-run the VLM or translation models.  All three required texts
(subject_answer_en, vlm_answer_en, concept_text) are already stored in the
JSON, so we only need to re-embed them with the same sentence-transformer.

Embeddings are batched by unique text to avoid redundant model calls.

Usage
-----
    python3.10 backfill_sac_vec.py results/vlm_results_qwen_20260227_152809.json
    # writes a .bak backup and overwrites the input file

    python3.10 backfill_sac_vec.py results/vlm_results_qwen_20260227_152809.json \\
        --output results/vlm_results_qwen_20260227_152809_v2.json
    # writes to a separate output file (input unchanged)
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np


# ── minimal Hebrew normaliser (mirrors src/utils_hebrew.py) ─────────────────
import unicodedata

_NIKKUD_RE = re.compile(r"[\u0591-\u05BD\u05BF\u05C1\u05C2\u05C4\u05C5\u05C7]")
_FINALS = str.maketrans("ךםןףץ", "כמנפצ")


def _normalize(text: str) -> str:
    text = _NIKKUD_RE.sub("", text or "")
    text = text.translate(_FINALS)
    return re.sub(r"\s+", " ", text).strip()


# ── vector-projection SAC (pure numpy, no torch needed here) ─────────────────

def _batch_sac_vec(
    embs_subj: np.ndarray,
    embs_vlm: np.ndarray,
    embs_concept: np.ndarray,
) -> np.ndarray:
    """For each row i compute cos(subj_i − proj_{concept_i}(subj_i),  vlm_i).

    All input arrays must be L2-normalised, shape [N, D].
    Returns float array of shape [N].
    """
    # Scalar projection of subj onto concept (= dot product for unit vectors)
    proj = np.einsum("nd,nd->n", embs_subj, embs_concept)          # [N]
    subj_perp = embs_subj - proj[:, None] * embs_concept             # [N, D]

    norms = np.linalg.norm(subj_perp, axis=1)                        # [N]
    # Avoid division by zero for subjects fully aligned with concept
    valid = norms > 1e-8
    norms_safe = np.where(valid, norms, 1.0)
    subj_perp_norm = subj_perp / norms_safe[:, None]
    subj_perp_norm[~valid] = 0.0                                      # zero → cos = 0

    return np.einsum("nd,nd->n", subj_perp_norm, embs_vlm)           # [N]


def _batch_sac_vec_both(
    embs_subj: np.ndarray,
    embs_vlm: np.ndarray,
    embs_concept: np.ndarray,
) -> np.ndarray:
    """For each row i compute cos(subj_perp_i, vlm_perp_i) wrt concept_i.

    All input arrays must be L2-normalised, shape [N, D].
    Returns float array of shape [N].
    """
    proj_subj = np.einsum("nd,nd->n", embs_subj, embs_concept)
    proj_vlm = np.einsum("nd,nd->n", embs_vlm, embs_concept)

    subj_perp = embs_subj - proj_subj[:, None] * embs_concept
    vlm_perp = embs_vlm - proj_vlm[:, None] * embs_concept

    subj_norms = np.linalg.norm(subj_perp, axis=1)
    vlm_norms = np.linalg.norm(vlm_perp, axis=1)

    subj_valid = subj_norms > 1e-8
    vlm_valid = vlm_norms > 1e-8
    both_valid = subj_valid & vlm_valid

    subj_safe = np.where(subj_valid, subj_norms, 1.0)
    vlm_safe = np.where(vlm_valid, vlm_norms, 1.0)

    subj_perp_norm = subj_perp / subj_safe[:, None]
    vlm_perp_norm = vlm_perp / vlm_safe[:, None]

    subj_perp_norm[~subj_valid] = 0.0
    vlm_perp_norm[~vlm_valid] = 0.0

    out = np.einsum("nd,nd->n", subj_perp_norm, vlm_perp_norm)
    out[~both_valid] = 0.0
    return out


# ── main ─────────────────────────────────────────────────────────────────────

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


def main():
    parser = argparse.ArgumentParser(
        description="Backfill sim_above_concept_vec into existing results JSON"
    )
    parser.add_argument("input", help="Path to results JSON")
    parser.add_argument(
        "--output", default=None,
        help="Output path (default: overwrite input, keeping a .bak backup)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"ERROR: file not found: {input_path}")

    # ── load ─────────────────────────────────────────────────────────────────
    print(f"Loading {input_path} ...")
    data = json.loads(input_path.read_text(encoding="utf-8"))
    all_results = data.get("results", [])

    full   = [r for r in all_results if r.get("subject_answer_en") and
              r.get("vlm_answer_en") and r.get("concept_text")]
    sparse = [r for r in all_results if r not in full]
    print(f"  {len(full)} fully-populated records, {len(sparse)} sparse records")

    if not full:
        sys.exit("No fully-populated records found — nothing to do.")

    # ── embed model ───────────────────────────────────────────────────────────
    embed_name = data.get(
        "embed",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    print(f"Loading embedding model: {embed_name} ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embed_name)
    print("[OK]\n")

    # ── batch-embed unique texts ──────────────────────────────────────────────
    unique_subj = list({_normalize(r["subject_answer_en"]) for r in full})
    unique_vlm  = list({_normalize(r["vlm_answer_en"])     for r in full})
    unique_conc = list({_normalize(r["concept_text"])      for r in full})
    all_texts   = unique_subj + unique_vlm + unique_conc

    print(f"Embedding {len(all_texts)} unique texts "
          f"({len(unique_subj)} subj, {len(unique_vlm)} vlm, "
          f"{len(unique_conc)} concepts) ...")
    raw_embs = model.encode(all_texts, show_progress_bar=True, batch_size=64)

    # L2-normalise once
    norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
    embs  = raw_embs / np.where(norms > 0, norms, 1.0)

    n_s, n_v = len(unique_subj), len(unique_vlm)
    emb_subj_map = {t: embs[i]           for i, t in enumerate(unique_subj)}
    emb_vlm_map  = {t: embs[n_s + i]     for i, t in enumerate(unique_vlm)}
    emb_conc_map = {t: embs[n_s + n_v + i] for i, t in enumerate(unique_conc)}

    # ── build batch arrays ────────────────────────────────────────────────────
    embs_subj = np.array([emb_subj_map[_normalize(r["subject_answer_en"])] for r in full])
    embs_vlm  = np.array([emb_vlm_map [_normalize(r["vlm_answer_en"])]     for r in full])
    embs_conc = np.array([emb_conc_map [_normalize(r["concept_text"])]     for r in full])

    sac_vecs = _batch_sac_vec(embs_subj, embs_vlm, embs_conc)
    sac_vecs_both = _batch_sac_vec_both(embs_subj, embs_vlm, embs_conc)

    # ── apply changes to full records ─────────────────────────────────────────
    for r, sv, svb in zip(full, sac_vecs, sac_vecs_both):
        # Rename scalar metric
        if "sim_above_concept" in r and "sim_above_concept_scalar" not in r:
            r["sim_above_concept_scalar"] = r.pop("sim_above_concept")
        elif "sim_above_concept" in r:
            r["sim_above_concept_scalar"] = r.pop("sim_above_concept")
        # Add vector metric
        r["sim_above_concept_vec"] = round(float(sv), 4)
        # Add bi-projected vector metric
        r["sim_above_concept_vec_both"] = round(float(svb), 4)

    # ── rename scalar in sparse records (no vec to compute) ──────────────────
    for r in sparse:
        if "sim_above_concept" in r and "sim_above_concept_scalar" not in r:
            r["sim_above_concept_scalar"] = r.pop("sim_above_concept")

    # ── reassemble in original order ─────────────────────────────────────────
    stem_map = {r["stem"]: r for r in full + sparse if "stem" in r}
    updated_results = [stem_map.get(r.get("stem"), r) for r in all_results]
    data["results"] = updated_results

    # ── output ────────────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    else:
        # Overwrite in-place, keep a backup
        bak_path = input_path.with_suffix(".bak.json")
        bak_path.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backup written to {bak_path}")
        out_path = input_path

    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"JSON written to {out_path}")

    # ── regenerate CSV ────────────────────────────────────────────────────────
    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in updated_results:
            writer.writerow({k: r.get(k, "") for k in CSV_COLUMNS})
    print(f"CSV  written to {csv_path}")

    # ── stats ─────────────────────────────────────────────────────────────────
    sv = sac_vecs
    svb = sac_vecs_both
    scalar_vals = np.array(
        [r["sim_above_concept_scalar"] for r in full if "sim_above_concept_scalar" in r],
        dtype=float,
    )
    print(
        f"\nsim_above_concept_scalar : mean={scalar_vals.mean():.4f}  "
        f"std={scalar_vals.std():.4f}  "
        f"min={scalar_vals.min():.4f}  max={scalar_vals.max():.4f}"
    )
    print(
        f"sim_above_concept_vec    : mean={sv.mean():.4f}  "
        f"std={sv.std():.4f}  "
        f"min={sv.min():.4f}  max={sv.max():.4f}"
    )
    print(
        f"sim_above_concept_vec_both: mean={svb.mean():.4f}  "
        f"std={svb.std():.4f}  "
        f"min={svb.min():.4f}  max={svb.max():.4f}"
    )
    print(f"\nDone - {len(full)} records updated.")


if __name__ == "__main__":
    main()
