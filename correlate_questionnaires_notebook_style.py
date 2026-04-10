from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce notebook-style questionnaire correlation table."
    )
    parser.add_argument(
        "--answers-file",
        default="results/vlm_results_qwen_20260227_152809.json",
        help="Answer-level results file (.json with 'results' key or .csv).",
    )
    parser.add_argument(
        "--questionnaire-1",
        default="questionnaire_1.csv",
        help="Path to questionnaire_1.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output folder for table exports.",
    )
    return parser.parse_args()


def parse_subject_id_from_label(label: str) -> int:
    m = re.search(r"(\d+)", str(label))
    if not m:
        raise ValueError(f"Could not parse subject id from {label!r}")
    return int(m.group(1))


def load_questionnaire(path: Path) -> pd.DataFrame:
    qt = pd.read_csv(path, index_col=0)
    qt.index = qt.index.astype(str).str.strip()
    qt.columns = ["pct_systems_thinking"]
    qt["subject_id"] = qt.index.str.extract(r"(\d+)", expand=False).astype(int)
    qt = qt.reset_index(drop=True)
    qt["pct_systems_thinking"] = pd.to_numeric(qt["pct_systems_thinking"], errors="coerce")
    qt = qt.dropna(subset=["pct_systems_thinking"]).copy()
    return qt[["subject_id", "pct_systems_thinking"]]


def load_answer_level(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data["results"] if isinstance(data, dict) and "results" in data else data
        df = pd.DataFrame(raw)
    else:
        df = pd.read_csv(path)

    needed = ["stem", "sim_above_concept_correct_vec_both"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in answer-level data: {missing}")

    df = df.copy()
    df["sim_above_concept_correct_vec_both"] = pd.to_numeric(
        df["sim_above_concept_correct_vec_both"], errors="coerce"
    )
    df = df.dropna(subset=["sim_above_concept_correct_vec_both", "stem"]).copy()
    df["subject_id"] = (
        df["stem"].astype(str).str.split("_").str[-1].map(parse_subject_id_from_label)
    )
    return df


def compute_corr_row(df: pd.DataFrame, x_col: str, y_col: str, name_col: str, name: str) -> dict:
    sub = df[[x_col, y_col]].dropna()
    r, p = stats.pearsonr(sub[x_col], sub[y_col])
    rho, p2 = stats.spearmanr(sub[x_col], sub[y_col])
    return {
        name_col: name,
        "pearson_r": round(float(r), 4),
        "pearson_p": round(float(p), 4),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p2), 4),
        "n": int(len(sub)),
    }


def sweep_weights(merged_weighted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for weight_correct in [i / 100 for i in range(0, 101)]:
        weight_similarity = 1.0 - weight_correct
        score = (
            weight_correct * merged_weighted["mean_is_correct_answer"]
            + weight_similarity * merged_weighted["mean_similarity"]
        )
        sub = pd.DataFrame(
            {
                "weighted_score": score,
                "pct_systems_thinking": merged_weighted["pct_systems_thinking"],
            }
        ).dropna()
        r, p = stats.pearsonr(sub["weighted_score"], sub["pct_systems_thinking"])
        rho, p2 = stats.spearmanr(sub["weighted_score"], sub["pct_systems_thinking"])
        rows.append(
            {
                "weight_correct": round(weight_correct, 2),
                "weight_similarity": round(weight_similarity, 2),
                "pearson_r": float(r),
                "pearson_p": float(p),
                "spearman_rho": float(rho),
                "spearman_p": float(p2),
                "n": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    answers_path = Path(args.answers_file)
    q1_path = Path(args.questionnaire_1)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_answer_level(answers_path)
    qt = load_questionnaire(q1_path)

    if "sim_above_concept_total_vec_both" not in df.columns:
        if "sim_above_concept_correct_vec_both" in df.columns:
            df["sim_above_concept_total_vec_both"] = df["sim_above_concept_correct_vec_both"]
        else:
            raise ValueError(
                "Missing both sim_above_concept_total_vec_both and sim_above_concept_correct_vec_both"
            )

    df["sim_above_concept_total_vec_both"] = pd.to_numeric(
        df["sim_above_concept_total_vec_both"], errors="coerce"
    )

    # Match notebook behavior: answer numbers 1/2 are considered correct.
    if "answer_num" in df.columns:
        answer_num_int = pd.to_numeric(df["answer_num"], errors="coerce")
    else:
        answer_num_int = pd.Series([float("nan")] * len(df), index=df.index)
    df["is_correct_answer"] = answer_num_int.isin([1, 2]).astype(float)
    lexicographic_bonus = float(df["sim_above_concept_total_vec_both"].max()) + 1e-6
    df["lexicographic_score_correct_first"] = (
        lexicographic_bonus * df["is_correct_answer"] + df["sim_above_concept_total_vec_both"]
    )
    df["correctness_plus_description_0p1"] = (
        df["is_correct_answer"] + 0.1 * df["sim_above_concept_total_vec_both"]
    )
    df["fair_score_correct_bonus_1p1"] = (
        1.1 * df["is_correct_answer"] + df["sim_above_concept_total_vec_both"]
    )
    df["weighted_score_50_50"] = (
        0.5 * df["is_correct_answer"] + 0.5 * df["sim_above_concept_total_vec_both"]
    )
    df["weighted_score_30_70"] = (
        0.3 * df["is_correct_answer"] + 0.7 * df["sim_above_concept_total_vec_both"]
    )

    vec_both_med_thr = float(df["sim_above_concept_correct_vec_both"].median())
    vec_both_avg_thr = float(df["sim_above_concept_correct_vec_both"].mean())

    per_subject = (
        df.groupby("subject_id")
        .agg(
            mean_SAC_vec_both=("sim_above_concept_correct_vec_both", "mean"),
            pct_high_vec_both_median=(
                "sim_above_concept_correct_vec_both",
                lambda x: (x >= vec_both_med_thr).mean() * 100,
            ),
            pct_high_vec_both_avg=(
                "sim_above_concept_correct_vec_both",
                lambda x: (x >= vec_both_avg_thr).mean() * 100,
            ),
        )
        .reset_index()
    )

    merged = per_subject.merge(qt, on="subject_id", how="inner")

    pipeline_metrics = [
        "mean_SAC_vec_both",
        "pct_high_vec_both_median",
        "pct_high_vec_both_avg",
    ]

    rows = []
    for pm in pipeline_metrics:
        sub = merged[[pm, "pct_systems_thinking"]].dropna()
        r, p = stats.pearsonr(sub[pm], sub["pct_systems_thinking"])
        rho, p2 = stats.spearmanr(sub[pm], sub["pct_systems_thinking"])
        rows.append(
            {
                "pipeline_metric": pm,
                "pearson_r": round(float(r), 4),
                "pearson_p": round(float(p), 4),
                "spearman_rho": round(float(rho), 4),
                "spearman_p": round(float(p2), 4),
                "n": int(len(sub)),
            }
        )

    corr_df = pd.DataFrame(rows).sort_values("spearman_rho", ascending=False).reset_index(drop=True)

    # Before vs after adding weights (subject-level, matched to questionnaire_1)
    weighted_subject = (
        df.groupby("subject_id")
        .agg(
            mean_SAC_correct_vec_both=("sim_above_concept_correct_vec_both", "mean"),
            lexicographic_score_correct_first=("lexicographic_score_correct_first", "mean"),
            correctness_plus_description_0p1=("correctness_plus_description_0p1", "mean"),
            fair_score_correct_bonus_1p1=("fair_score_correct_bonus_1p1", "mean"),
            mean_is_correct_answer=("is_correct_answer", "mean"),
            mean_similarity=("sim_above_concept_total_vec_both", "mean"),
            weighted_score_50_50=("weighted_score_50_50", "mean"),
            weighted_score_30_70=("weighted_score_30_70", "mean"),
        )
        .reset_index()
    )
    merged_weighted = weighted_subject.merge(qt, on="subject_id", how="inner")

    weighted_rows = [
        compute_corr_row(
            merged_weighted,
            x_col="mean_SAC_correct_vec_both",
            y_col="pct_systems_thinking",
            name_col="pipeline_metric",
            name="mean_SAC_correct_vec_both",
        ),
        compute_corr_row(
            merged_weighted,
            x_col="lexicographic_score_correct_first",
            y_col="pct_systems_thinking",
            name_col="pipeline_metric",
            name="lexicographic_score_correct_first",
        ),
        compute_corr_row(
            merged_weighted,
            x_col="correctness_plus_description_0p1",
            y_col="pct_systems_thinking",
            name_col="pipeline_metric",
            name="correctness_plus_description_0p1",
        ),
        compute_corr_row(
            merged_weighted,
            x_col="fair_score_correct_bonus_1p1",
            y_col="pct_systems_thinking",
            name_col="pipeline_metric",
            name="fair_score_correct_bonus_1p1",
        ),
        compute_corr_row(
            merged_weighted,
            x_col="weighted_score_50_50",
            y_col="pct_systems_thinking",
            name_col="pipeline_metric",
            name="weighted_score_50_50",
        ),
        compute_corr_row(
            merged_weighted,
            x_col="weighted_score_30_70",
            y_col="pct_systems_thinking",
            name_col="pipeline_metric",
            name="weighted_score_30_70",
        ),
    ]
    weighted_corr_df = pd.DataFrame(weighted_rows)

    baseline = weighted_corr_df.loc[
        weighted_corr_df["pipeline_metric"] == "mean_SAC_correct_vec_both"
    ].iloc[0]
    delta_rows = []
    for metric in [
        "lexicographic_score_correct_first",
        "correctness_plus_description_0p1",
        "fair_score_correct_bonus_1p1",
        "weighted_score_50_50",
        "weighted_score_30_70",
    ]:
        row = weighted_corr_df.loc[weighted_corr_df["pipeline_metric"] == metric].iloc[0]
        delta_rows.append(
            {
                "weighted_metric": metric,
                "delta_pearson_r": round(float(row["pearson_r"] - baseline["pearson_r"]), 6),
                "delta_spearman_rho": round(
                    float(row["spearman_rho"] - baseline["spearman_rho"]), 6
                ),
                "delta_pearson_p": round(float(row["pearson_p"] - baseline["pearson_p"]), 6),
                "delta_spearman_p": round(
                    float(row["spearman_p"] - baseline["spearman_p"]), 6
                ),
            }
        )
    delta_df = pd.DataFrame(delta_rows)

    sweep_df = sweep_weights(merged_weighted)
    best_spearman = sweep_df.sort_values(["spearman_rho", "pearson_r"], ascending=False).iloc[0]
    best_pearson = sweep_df.sort_values(["pearson_r", "spearman_rho"], ascending=False).iloc[0]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"questionnaires_correlation_notebook_style_{stamp}.csv"
    out_weighted_csv = out_dir / f"questionnaires_correlation_weighted_compare_{stamp}.csv"
    out_delta_csv = out_dir / f"questionnaires_correlation_weighted_deltas_{stamp}.csv"
    out_sweep_csv = out_dir / f"questionnaires_correlation_weight_sweep_{stamp}.csv"
    out_md = out_dir / f"questionnaires_correlation_notebook_style_{stamp}.md"

    corr_df.to_csv(out_csv, index=False)
    weighted_corr_df.to_csv(out_weighted_csv, index=False)
    delta_df.to_csv(out_delta_csv, index=False)
    sweep_df.to_csv(out_sweep_csv, index=False)

    table_text = corr_df.to_string(index=False)
    weighted_table_text = weighted_corr_df.to_string(index=False)
    delta_table_text = delta_df.to_string(index=False)
    sweep_table_text = sweep_df.sort_values(["spearman_rho", "pearson_r"], ascending=False).to_string(index=False)

    md_lines = [
        "# Notebook-Style Correlation Table",
        "",
        f"- answers_file: {answers_path}",
        f"- questionnaire_1: {q1_path}",
        f"- matched subjects: {len(merged)}",
        f"- matched subjects (weights section): {len(merged_weighted)}",
        f"- vec_both median threshold: {vec_both_med_thr:.4f}",
        f"- vec_both mean threshold: {vec_both_avg_thr:.4f}",
        "",
        "## Notebook-Style Table",
        "```text",
        table_text,
        "```",
        "",
        "## Before vs After Adding Weights",
        "```text",
        weighted_table_text,
        "```",
        "",
        "## Deltas vs Unweighted Baseline (mean_SAC_correct_vec_both)",
        "```text",
        delta_table_text,
        "```",
        "",
        "## Weight Sweep (0.00 = all similarity, 1.00 = all correctness)",
        "```text",
        sweep_table_text,
        "```",
        "",
        "## Best Weights",
        f"- Best Spearman rho: weight_correct={best_spearman['weight_correct']:.2f}, weight_similarity={best_spearman['weight_similarity']:.2f}, rho={best_spearman['spearman_rho']:.4f}, pearson={best_spearman['pearson_r']:.4f}",
        f"- Best Pearson r: weight_correct={best_pearson['weight_correct']:.2f}, weight_similarity={best_pearson['weight_similarity']:.2f}, r={best_pearson['pearson_r']:.4f}, rho={best_pearson['spearman_rho']:.4f}",
        "",
        "## Output Files",
        f"- notebook-style CSV: {out_csv}",
        f"- weighted comparison CSV: {out_weighted_csv}",
        f"- weighted deltas CSV: {out_delta_csv}",
        f"- weight sweep CSV: {out_sweep_csv}",
    ]
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print("Notebook-style table:")
    print(corr_df.to_string(index=False))
    print("\nBefore vs after weights:")
    print(weighted_corr_df.to_string(index=False))
    print("\nDeltas vs unweighted baseline:")
    print(delta_df.to_string(index=False))
    print("\nWeight sweep (top 10 by Spearman then Pearson):")
    print(sweep_df.sort_values(["spearman_rho", "pearson_r"], ascending=False).head(10).to_string(index=False))
    print(
        f"\nBest Spearman weight: correct={best_spearman['weight_correct']:.2f}, "
        f"similarity={best_spearman['weight_similarity']:.2f}, "
        f"rho={best_spearman['spearman_rho']:.4f}, r={best_spearman['pearson_r']:.4f}"
    )
    print(
        f"Best Pearson weight: correct={best_pearson['weight_correct']:.2f}, "
        f"similarity={best_pearson['weight_similarity']:.2f}, "
        f"r={best_pearson['pearson_r']:.4f}, rho={best_pearson['spearman_rho']:.4f}"
    )
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_weighted_csv}")
    print(f"Saved: {out_delta_csv}")
    print(f"Saved: {out_sweep_csv}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()