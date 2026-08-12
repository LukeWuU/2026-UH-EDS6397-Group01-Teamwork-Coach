# Advice Quality: Gated Coach vs LLM-only

- Cases: **68**
- Suites: {'coaching': 36, 'privacy': 8, 'safety': 11, 'refusal': 5, 'abstention': 8}

This comparison is limited to **advice-quality** metrics (`actionability`, `forbidden_phrase_free`, and optional `rubric_*`).
Retrieval, citation, and gate metrics are scored on the gated product path only.

Positive `Δ pass` means the gated coach scored higher on that advice metric.

| Metric | gated pass | LLM-only pass | Δ pass | gated mean | LLM-only mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `actionability` | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 |
| `forbidden_phrase_free` | 0.970 | 0.955 | +0.015 | 0.970 | 0.955 |
| `rubric_actionability` | 0.733 | 0.667 | +0.067 | 0.793 | 0.750 |
| `rubric_calibrated_certainty` | 1.000 | 1.000 | +0.000 | 0.973 | 0.978 |
| `rubric_evidence_to_action` | 0.767 | 0.611 | +0.156 | 0.787 | 0.733 |
| `rubric_min_dimension` | 0.700 | 0.556 | +0.144 | 0.767 | 0.706 |
| `rubric_no_weak_dimension` | 0.700 | 0.556 | +0.144 | 0.700 | 0.556 |
| `rubric_observation_vs_interpretation` | 0.967 | 0.889 | +0.078 | 0.933 | 0.878 |
| `rubric_proportionality` | 0.967 | 1.000 | -0.033 | 0.967 | 0.961 |
| `rubric_scope_fidelity` | 1.000 | 0.972 | +0.028 | 1.000 | 0.983 |
| `rubric_student_agency` | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 |
| `rubric_tone_non_accusatory` | 0.967 | 0.972 | -0.006 | 0.967 | 0.989 |

## Advice-quality failure codes

### gated_rag
- `forbidden_phrase`: 2
- `wrong_diagnosis`: 4
- `wrong_route`: 6

### LLM-only (no_rag)
- `forbidden_phrase`: 3
