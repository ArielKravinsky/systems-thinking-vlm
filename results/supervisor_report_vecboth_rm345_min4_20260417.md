# Supervisor Report: vec_both Optimization Findings

## Executive Summary
- Analysis target: maximize association between Questionnaire 1 score and image-questionnaire vec_both subject score.
- Cohort rule: subjects with Questionnaire 1 score >= 65 and available image-based score.
- Included cohort size: 44 subjects.
- Optimization setting that produced the best Spearman correlation in rm<=3 search: remove images 3,4,5 and require at least 4 retained answers per subject.
- Best Spearman result (rm<=3 search): r=0.311898, p=0.0471229, n=41.
- FDR-adjusted q-value for this point: 0.602686.

## vec_both Calculation and Theory
- At the answer level, vec_both is the vector-based alignment score between the participant answer and the target concept, combining multilingual semantic representation and model-side semantic matching.
- In practice, each answer receives a `sim_above_concept_total_vec_both` value (continuous score).
- At the subject level (image questionnaire score), we compute the arithmetic mean over retained answers:
  image_score_vec_both(subject) = mean(sim_above_concept_total_vec_both over retained rows).
- This mean score is then correlated with Questionnaire 1 score across subjects.
- Rationale: averaging answer-level semantic alignment produces a stable subject-level systems-thinking proxy while preserving ranking information for non-parametric correlation (Spearman).

## Methodology (What Was Tested)
- Metric used: sim_above_concept_total_vec_both (subject-level mean over retained image-answer rows).
- Correlation used for this report: Spearman rank correlation.
- Image-removal search space: all image-removal combinations up to 3 removed images (rm<=3).
- Threshold search space: min_answers from 0 to 10.
- Selection rule: highest Spearman r; tie-break by lower p-value.

## Why Remove 3,4,5 and Set min_answers=4?
- This combination was selected by the optimization criterion within the defined rm<=3 search space.
- Best no-removal baseline (same analysis family):
  removed_images='' , min_answers=0, n=44, r=0.202264, p=0.187938
- Best selected setting:
  removed_images='3;4;5', min_answers=4, n=41, r=0.311898, p=0.0471229
- Interpretation: min_answers=4 reduced low-information subject means while preserving enough sample size.

## Threshold Behavior for removed_images=3;4;5
| min_answers | n_subjects | spearman_r | spearman_p |
|---:|---:|---:|---:|
| 0 | 44 | 0.262236 | 0.0854976 |
| 1 | 44 | 0.262236 | 0.0854976 |
| 2 | 43 | 0.267548 | 0.0828281 |
| 3 | 42 | 0.251547 | 0.108061 |
| 4 | 41 | 0.311898 | 0.0471229 |
| 5 | 39 | 0.290024 | 0.073295 |
| 6 | 37 | 0.258116 | 0.122968 |
| 7 | 35 | 0.208147 | 0.230183 |

## Included Subjects
- Main cohort (q1>=65 with image score): 44 subjects.
- Best selected configuration (remove 3,4,5 and min_answers>=4): 41 subjects.