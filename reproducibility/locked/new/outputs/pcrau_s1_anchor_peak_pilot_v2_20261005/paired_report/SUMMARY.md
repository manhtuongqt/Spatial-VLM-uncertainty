# C1/P1/P2 paired report — 05/10/2026

**Pilot complete; upgrade gates FAIL. No verifier/calibration/IID.**

| Arm | Selected epoch | Epochs / steps | Anchor /61 | Swap /15 | FOUND-both /16 |
|---|---:|---:|---:|---:|---:|
| M0 | — | — | 44 | 6 | 11 |
| C1 | 11 | 15 / 240 | 48 | 8 | 12 |
| P1 | 8 | 13 / 208 | 48 | 8 | 12 |
| P2 | 1 | 6 / 96 | 44 | 6 | 11 |

P1−C1: no dev correction or regression; no added binding benefit demonstrated.
P1−P2: +4 hits, +2 pairs; all four corrected cases concern lemon in two families.
P1 remains below49/61. P1 improves3/5 train-empty peaks vsC1, worsens2/5;
P1 still increases3/5 vsM0. On dev empties P1 improvesall3 vsM0 but only1/3 vsC1.

| Gate | Pass |
|---|---|
| anchor_hit_ge_49_of_61 | False |
| empty_each_no_increase_1e_6 | True |
| found_both_ge_12_of_16 | True |
| latency_le_20_percent | True |
| peak_objective_vs_C1 | False |
| phrase_specific_vs_P2 | True |
| swap_both_ge_8_of_15 | True |
| technical | True |

![Learning curves](learning_curves.png)

Raw train objectives differ between C1 and P1/P2; their magnitudes are not directly comparable.
Family bootstrap is conditional on dev-selected checkpoints. P1−P2 hit delta CI includes0;
P1−C1 observed hit/pair labels coincide, so bootstrap CI[0,0] does not prove population equivalence.

See [family deltas](family_paired_deltas.csv), [each empty case](empty_cases.csv),
[13 remaining misses](remaining_dev_misses.jsonl), [full summary](../summary.json)
and [Vietnamese report](../../../../plan/S1_ANCHOR_PEAK_PILOT_V2_20261005.md).
