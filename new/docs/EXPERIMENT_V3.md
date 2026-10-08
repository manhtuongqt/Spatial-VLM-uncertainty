# V3 predeclared experiment

V2 epoch 13 / step 1120 remains the immutable baseline.  V3 is a new run; it
must never resume from the V2 optimizer state.

## Changes fixed before training

- dropout 0.20 and one-modality feature dropout probability 0.15;
- AdamW learning rate 1e-4 and weight decay 5e-4;
- three-epoch early stopping patience;
- one extra presentation per epoch for families containing `ABSENT`,
  `between_in_depth`, or `left_of` (92 extra presentations, using the maximum
  repeat when criteria overlap);
- reduced positive-class pressure plus explicit negative penalties for
  semantic and occlusion source logits;
- checkpoint every three epochs plus an independently updated best checkpoint;
- best checkpoint eligibility requires dev grounding accuracy >= 0.97;
- seeds 24082026, 24082027, and 24082028.

Each epoch contains 412 family presentations / 2,060 sample presentations in
103 family-complete batches.  The original 320 families remain present once;
oversampling adds repetitions and does not redefine the locked dataset split.

## Success criteria

- evaluator target-mask grounding remains at least 0.97;
- answerability macro-F1 target at least 0.85;
- ABSENT F1 target at least 0.72;
- five-source macro-F1 target at least 0.72;
- semantic and occlusion source F1 target at least 0.62;
- post-freeze risk coverage at 5% target at least 0.35.

These are development targets, not Test-IID/Test-OOD claims.  Source thresholds
are fit only after checkpoint freeze on the calibration archive and must be
reported as calibration-fit parameters, not calibration-set generalization.
