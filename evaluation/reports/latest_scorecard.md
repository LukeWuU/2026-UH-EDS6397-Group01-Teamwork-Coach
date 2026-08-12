# Teamwork Coach — Evaluation Scorecard

**Overall readiness: CONDITIONAL**

- System: `gated_rag`
- Cases scored: **68**
- Suites: {'coaching': 36, 'privacy': 8, 'safety': 11, 'refusal': 5, 'abstention': 8}
- Generated (UTC): 2026-08-12T17:03:25Z
- Note: Scorecard based on n=68 scored case(s).

## Readiness notes

- Suite(s) in watch band (70–90%): `abstention`, `coaching`

## Key gates (PRD acceptance)

| Gate | n | pass_rate | threshold | status |
| --- | ---: | ---: | ---: | --- |
| `gate_integrity` | 32 | 1.000 | 1.00 | **pass** |
| `citation_present` | 32 | 1.000 | 0.95 | **pass** |
| `citation_from_retrieved` | 32 | 1.000 | 0.95 | **pass** |
| `pii_leakage_free` | 68 | 1.000 | 1.00 | **pass** |
| `high_risk_match` | 16 | 1.000 | 1.00 | **pass** |

## Suite rollup

| Suite | n | case pass rate | status |
| --- | ---: | ---: | --- |
| `abstention` | 8 | 0.750 | **watch** |
| `coaching` | 36 | 0.778 | **watch** |
| `privacy` | 8 | 1.000 | **pass** |
| `refusal` | 5 | 1.000 | **pass** |
| `safety` | 11 | 1.000 | **pass** |

## Headline metrics

| Metric | n | mean | pass_rate |
| --- | ---: | ---: | ---: |
| `route_match` | 68 | 0.912 | 0.912 |
| `citation_present` | 32 | 1.000 | 1.000 |
| `citation_from_retrieved` | 32 | 1.000 | 1.000 |
| `gate_integrity` | 32 | 1.000 | 1.000 |
| `high_risk_match` | 16 | 1.000 | 1.000 |
| `pii_leakage_free` | 68 | 1.000 | 1.000 |
| `forbidden_phrase_free` | 66 | 0.970 | 0.970 |
| `diagnosis_primary_hit` | 35 | 0.886 | 0.886 |
| `actionability` | 30 | 1.000 | 1.000 |

## Advice quality vs LLM-only (no retrieval)

Compared only on advice-quality metrics. Citation, retrieval, and gate scores are product-path checks and are not used against the LLM-only baseline.

| Metric | gated pass | LLM-only pass | Δ pass |
| --- | ---: | ---: | ---: |
| `actionability` | 1.000 | 1.000 | +0.000 |
| `forbidden_phrase_free` | 0.970 | 0.955 | +0.015 |
| `rubric_actionability` | 0.733 | 0.667 | +0.067 |
| `rubric_calibrated_certainty` | 1.000 | 1.000 | +0.000 |
| `rubric_evidence_to_action` | 0.767 | 0.611 | +0.156 |
| `rubric_min_dimension` | 0.700 | 0.556 | +0.144 |
| `rubric_no_weak_dimension` | 0.700 | 0.556 | +0.144 |
| `rubric_observation_vs_interpretation` | 0.967 | 0.889 | +0.078 |
| `rubric_proportionality` | 0.967 | 1.000 | -0.033 |
| `rubric_scope_fidelity` | 1.000 | 0.972 | +0.028 |
| `rubric_student_agency` | 1.000 | 1.000 | +0.000 |
| `rubric_tone_non_accusatory` | 0.967 | 0.972 | -0.006 |

## Top failure codes

- `wrong_route`: 6
- `wrong_diagnosis`: 4
- `forbidden_phrase`: 2
