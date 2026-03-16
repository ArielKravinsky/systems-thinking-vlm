"""
compare_gold.py  –  Post-hoc comparison of VLM answers and subject answers
                    against manually-authored gold answers.

Usage:
    python compare_gold.py [results_json] [--gold-dir dataset/gold_answers]

Outputs:
    <stem>_gold_comparison.csv  – per-sample metrics vs gold
    <stem>_gold_summary.txt     – aggregated stats per question / overall
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util
from bert_score import BERTScorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


HE_EN_MODEL = "Helsinki-NLP/opus-mt-tc-big-he-en"
EMBED_MODEL  = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LABSE_MODEL  = "sentence-transformers/LaBSE"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_gold_answers(gold_dir: Path) -> dict:
    """Returns {'{q}_{a}': '<hebrew text>'} for every .txt in gold_dir."""
    gold = {}
    for f in gold_dir.glob("*.txt"):
        key = f.stem           # e.g. "1_2"
        gold[key] = f.read_text(encoding="utf-8").strip()
    return gold


def embed_cos(model, a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    emb = model.encode([a, b], convert_to_tensor=True)
    return float(util.cos_sim(emb[0], emb[1]).item())


def bert_score_pair(scorer, hyp: str, ref: str):
    if not hyp.strip() or not ref.strip():
        return 0.0, 0.0, 0.0
    P, R, F = scorer.score([hyp], [ref])
    return round(float(P[0]), 4), round(float(R[0]), 4), round(float(F[0]), 4)


def translate(text: str, tokenizer, model, device: str) -> str:
    if not text.strip():
        return ""
    inputs = tokenizer(text, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare answers against gold standard")
    parser.add_argument("results_json", nargs="?",
                        help="Path to vlm_results_*.json; defaults to the newest one")
    parser.add_argument("--gold-dir", default="dataset/gold_answers",
                        help="Directory containing gold answer .txt files")
    args = parser.parse_args()

    # ── resolve files ────────────────────────────────────────────────────────
    if args.results_json:
        results_path = Path(args.results_json)
    else:
        candidates = sorted(Path(".").glob("vlm_results_*.json"))
        if not candidates:
            raise FileNotFoundError("No vlm_results_*.json found in current directory")
        results_path = candidates[-1]
        print(f"Using newest results file: {results_path}")

    gold_dir = Path(args.gold_dir)
    if not gold_dir.exists():
        raise FileNotFoundError(f"Gold answers directory not found: {gold_dir}")

    # ── load data ────────────────────────────────────────────────────────────
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]
    print(f"Loaded {len(results)} results from {results_path}")

    gold = load_gold_answers(gold_dir)
    print(f"Loaded {len(gold)} gold answers from {gold_dir}")

    # ── load models ──────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    print("Loading mpnet embedding model...")
    mpnet = SentenceTransformer(EMBED_MODEL, device=device)
    print("Loading LaBSE embedding model...")
    labse = SentenceTransformer(LABSE_MODEL, device=device)
    print("Loading BERTScorer...")
    scorer = BERTScorer(lang="en", rescale_with_baseline=False, device="cpu")
    print("Loading He→En translation model...")
    he_en_tok = AutoTokenizer.from_pretrained(HE_EN_MODEL)
    he_en_mdl = AutoModelForSeq2SeqLM.from_pretrained(HE_EN_MODEL)
    if device == "cuda":
        he_en_mdl = he_en_mdl.to(device)
    print("All models ready.\n")

    # ── output files ─────────────────────────────────────────────────────────
    stem = results_path.stem
    out_csv  = results_path.parent / f"{stem}_gold_comparison.csv"
    out_txt  = results_path.parent / f"{stem}_gold_summary.txt"
    out_json = results_path.parent / f"{stem}_gold_comparison.json"

    COLUMNS = [
        "stem", "question_num", "answer_num", "participant_num",
        "image_key",
        "gold_he", "gold_en",
        # VLM vs gold
        "vlm_answer_he", "vlm_answer_en",
        "vlm_vs_gold_mpnet_he", "vlm_vs_gold_labse_he",
        "vlm_vs_gold_mpnet_en", "vlm_vs_gold_bs_f1",
        # subject vs gold
        "subject_answer_he", "subject_answer_en",
        "subj_vs_gold_mpnet_he", "subj_vs_gold_labse_he",
        "subj_vs_gold_mpnet_en", "subj_vs_gold_bs_f1",
        # VLM vs subject (already in results, carry over)
        "vlm_vs_subj_mpnet_he",
    ]

    rows = []
    skipped = 0

    for r in tqdm(results, desc="Comparing to gold", unit="sample"):
        q  = r["question_num"]
        a  = r["answer_num"]
        image_key = f"{q}_{a}"

        if image_key not in gold:
            skipped += 1
            continue

        gold_he = gold[image_key]
        gold_en = translate(gold_he, he_en_tok, he_en_mdl, device)

        vlm_he  = r.get("vlm_answer_he", "")
        vlm_en  = r.get("vlm_answer_en", "")
        subj_he = r.get("subject_answer_he", "")
        # translate subject to English if not already stored
        subj_en = r.get("subject_answer_en", "") or translate(subj_he, he_en_tok, he_en_mdl, device)

        # VLM vs gold
        vlm_mpnet_he = embed_cos(mpnet, vlm_he,  gold_he)
        vlm_labse_he = embed_cos(labse, vlm_he,  gold_he)
        vlm_mpnet_en = embed_cos(mpnet, vlm_en,  gold_en)
        _, _, vlm_bs = bert_score_pair(scorer, vlm_en, gold_en)

        # subject vs gold
        sub_mpnet_he = embed_cos(mpnet, subj_he, gold_he)
        sub_labse_he = embed_cos(labse, subj_he, gold_he)
        sub_mpnet_en = embed_cos(mpnet, subj_en, gold_en)
        _, _, sub_bs = bert_score_pair(scorer, subj_en, gold_en)

        rows.append({
            "stem": r["stem"],
            "question_num": q,
            "answer_num": a,
            "participant_num": r.get("participant_num", ""),
            "image_key": image_key,
            "gold_he": gold_he,
            "gold_en": gold_en,
            "vlm_answer_he": vlm_he,
            "vlm_answer_en": vlm_en,
            "vlm_vs_gold_mpnet_he": round(vlm_mpnet_he, 4),
            "vlm_vs_gold_labse_he": round(vlm_labse_he, 4),
            "vlm_vs_gold_mpnet_en": round(vlm_mpnet_en, 4),
            "vlm_vs_gold_bs_f1":    round(vlm_bs,       4),
            "subject_answer_he": subj_he,
            "subject_answer_en": subj_en,
            "subj_vs_gold_mpnet_he": round(sub_mpnet_he, 4),
            "subj_vs_gold_labse_he": round(sub_labse_he, 4),
            "subj_vs_gold_mpnet_en": round(sub_mpnet_en, 4),
            "subj_vs_gold_bs_f1":    round(sub_bs,       4),
            "vlm_vs_subj_mpnet_he": round(r.get("similarity", 0.0), 4),
        })

    print(f"\nProcessed {len(rows)} samples ({skipped} skipped – no gold answer)")

    # ── write CSV ─────────────────────────────────────────────────────────────
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")

    # ── aggregate summary ─────────────────────────────────────────────────────
    def mean(vals): return sum(vals) / len(vals) if vals else 0.0

    metrics_vlm  = ["vlm_vs_gold_mpnet_he", "vlm_vs_gold_labse_he", "vlm_vs_gold_mpnet_en", "vlm_vs_gold_bs_f1"]
    metrics_subj = ["subj_vs_gold_mpnet_he","subj_vs_gold_labse_he","subj_vs_gold_mpnet_en","subj_vs_gold_bs_f1"]

    lines = []
    lines.append(f"Gold Comparison Summary – {results_path.name}")
    lines.append("=" * 70)
    lines.append(f"{'Metric':<30} {'VLM vs Gold':>14} {'Subj vs Gold':>14} {'Δ (VLM-Subj)':>14}")
    lines.append("-" * 70)

    for mv, ms in zip(metrics_vlm, metrics_subj):
        v_vals = [r[mv] for r in rows]
        s_vals = [r[ms] for r in rows]
        mv_mean = mean(v_vals)
        ms_mean = mean(s_vals)
        lines.append(f"{mv:<30} {mv_mean:>14.4f} {ms_mean:>14.4f} {mv_mean-ms_mean:>+14.4f}")

    lines.append("=" * 70)
    lines.append("\nPer-question breakdown (VLM vs Gold  |  Subject vs Gold)  [mpnet_he]")
    lines.append("-" * 70)

    from collections import defaultdict
    qgroups = defaultdict(list)
    for r in rows:
        qgroups[r["question_num"]].append(r)

    for qnum in sorted(qgroups, key=lambda x: int(x)):
        qrows = qgroups[qnum]
        v = mean([r["vlm_vs_gold_mpnet_he"]  for r in qrows])
        s = mean([r["subj_vs_gold_mpnet_he"] for r in qrows])
        lines.append(f"  Q{qnum:<4}  VLM={v:.4f}  Subj={s:.4f}  Δ={v-s:+.4f}  (n={len(qrows)})")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    out_txt.write_text(summary_text + "\n", encoding="utf-8")
    print(f"\nWrote {out_txt}")

    # ── write JSON ────────────────────────────────────────────────────────────
    from collections import defaultdict
    qgroups2 = defaultdict(list)
    for r in rows:
        qgroups2[r["question_num"]].append(r)

    per_question = {}
    for q, qrows in sorted(qgroups2.items(), key=lambda x: int(x[0])):
        per_question[f"Q{q}"] = {
            "n": len(qrows),
            "vlm_vs_gold_mpnet_he":  mean([r["vlm_vs_gold_mpnet_he"]  for r in qrows]),
            "vlm_vs_gold_labse_he":  mean([r["vlm_vs_gold_labse_he"]  for r in qrows]),
            "vlm_vs_gold_mpnet_en":  mean([r["vlm_vs_gold_mpnet_en"]  for r in qrows]),
            "vlm_vs_gold_bs_f1":     mean([r["vlm_vs_gold_bs_f1"]     for r in qrows]),
            "subj_vs_gold_mpnet_he": mean([r["subj_vs_gold_mpnet_he"] for r in qrows]),
            "subj_vs_gold_labse_he": mean([r["subj_vs_gold_labse_he"] for r in qrows]),
            "subj_vs_gold_mpnet_en": mean([r["subj_vs_gold_mpnet_en"] for r in qrows]),
            "subj_vs_gold_bs_f1":    mean([r["subj_vs_gold_bs_f1"]    for r in qrows]),
        }

    json_output = {
        "source_file": results_path.name,
        "gold_answers_dir": str(gold_dir),
        "n_samples": len(rows),
        "overall": {
            "vlm_vs_gold_mpnet_he":  mean([r["vlm_vs_gold_mpnet_he"]  for r in rows]),
            "vlm_vs_gold_labse_he":  mean([r["vlm_vs_gold_labse_he"]  for r in rows]),
            "vlm_vs_gold_mpnet_en":  mean([r["vlm_vs_gold_mpnet_en"]  for r in rows]),
            "vlm_vs_gold_bs_f1":     mean([r["vlm_vs_gold_bs_f1"]     for r in rows]),
            "subj_vs_gold_mpnet_he": mean([r["subj_vs_gold_mpnet_he"] for r in rows]),
            "subj_vs_gold_labse_he": mean([r["subj_vs_gold_labse_he"] for r in rows]),
            "subj_vs_gold_mpnet_en": mean([r["subj_vs_gold_mpnet_en"] for r in rows]),
            "subj_vs_gold_bs_f1":    mean([r["subj_vs_gold_bs_f1"]    for r in rows]),
        },
        "per_question": per_question,
        "samples": rows,
    }
    import json as _json
    out_json.write_text(_json.dumps(json_output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
