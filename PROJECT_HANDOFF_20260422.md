# Project Handoff (2026-04-22)

## Locked Analysis Configuration
- Questionnaire 1 pass threshold: q1 >= 65
- Metric: sim_above_concept_total_vec_both
- Correlation: Spearman
- Agreed reference result under n_removed<=3:
  - removed_images = 3;4;5
  - min_answers = 4
  - n = 41
  - r = 0.311898
  - p = 0.047123

## Key Files to Continue From
- results/maxcorr_q1ge65_vecboth_spearman_rm0_10_all_20260422.csv
- results/maxcorr_q1ge65_vecboth_spearman_top5_cutoff_combinations_20260422.csv
- results/maxcorr_q1ge65_vecboth_spearman_top5_cutoff_combinations_conservative_rm_le_3_20260422.csv
- results/maxcorr_q1ge65_vecboth_spearman_top5_cutoff_combinations_rm_lt_3_20260422.csv
- results/included_subjects_q1ge65_with_image_vecboth_20260417.csv
- results/included_subjects_best_rm3_345_min4_vecboth_20260417.csv
- results/supervisor_report_vecboth_rm345_min4_20260417_no_caution_with_theory.pdf

## Continue on Another Computer
1. Clone repo and checkout branch used for this commit.
2. Open workspace root in VS Code.
3. Start from files above and keep the locked configuration unless changed explicitly.

## Notes
- Conversation history itself is managed by the chat client/account and cannot be exported/deleted from this repo by script.
