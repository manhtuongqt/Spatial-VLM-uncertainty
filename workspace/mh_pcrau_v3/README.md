# MH-PCRA-U-v3 code workspace

This directory is the canonical location for new v3 implementation code from Day 3 onward.
It follows the frozen architecture in plan/plan_spatial_vlm_multihead_pcra_u_v3_exact.md.

## Layout

- hidden_state_adapter.py: exact byte copy of the passing Day 2 adapter.
- tests/: workspace tests; import the workspace adapter.
- g0_locked/: copies of the original G0 runner/finalizer/test and two failed runner versions.
  These scripts contain source-location-dependent paths and are retained for
  provenance, not for rerunning a completed gate.
- g1/: development-only schema and split audit code.
  It now also contains the label provenance/coverage packet builder,
  Python desktop reviewer and returned-human-review validator.
- multihead_v3.py: exact shared trunk and seven-head Day-4 implementation.
- loss_v3.py: strict masked S1a/S1b multi-task objectives.
- day4_smoke.py: reproducible tests and Day-4 evidence materializer.
- day5_prepare.py: append-only Day-5 selection, supervision and run lock.
- day5_cache.py: real RoboRefer h_spatial cache extraction under G0 precision.
- day5_train.py: six-candidate gradient smoke and retained attempt-01 overfit.
- day5_prepare_r2.py / day5_train_r2.py: preregistered variance remediation.
- plot_day5.py: Matplotlib figures generated only from recorded Day-5 evidence.
- day6_prepare.py: family-disjoint S1a input/supervision audit and base run lock.
- day6_cache.py: 320-family real h_spatial cache with verified Day-5 reuse.
- day6_prepare_r2.py / day6_train.py: deterministic S1a source lock and training.
- plot_day6.py: Matplotlib figures from recorded S1a logs and metrics.

The original Day 2 files under RoboRefer/ remain at their locked paths so the
G0 run-lock hashes and result manifest continue to verify. Future v3 code goes
here; any RoboRefer integration should use an explicit import bridge rather
than silently changing the frozen G0 sources. Data, locks and evidence remain
under datasets/, protocol/ and ketqua1/ respectively.

Run workspace tests from the project root with the isolated interpreter:

    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 .conda-roborefer/bin/python -m unittest discover -s workspace/mh_pcrau_v3/tests -p 'test_*.py'

Initial G1 inventory:

    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 .conda-roborefer/bin/python -m workspace.mh_pcrau_v3.g1.audit_development --output ketqua1/01_dau_vao_tien_xu_ly/ngay_03/INITIAL_SCHEMA_INVENTORY.json

The inventory runner writes exclusively; use a new result filename for a new
attempt. The present output has artifact status RUN_COMPLETE and gate state
G1_IN_PROGRESS; it is not a G1 gate decision.

Build the pilot human-review packet with:

    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 .conda-roborefer/bin/python -m workspace.mh_pcrau_v3.g1.build_label_audit

The builder refuses to overwrite an existing packet. Its output and reviewer
instructions are in ketqua1/01_dau_vao_tien_xu_ly/ngay_03/label_audit_packet_v1/.
Open the desktop reviewer with `python -m workspace.mh_pcrau_v3.g1.human_review_gui`;
returned CSVs can be checked with g1/validate_human_review.py. None of these
tools approves G1 or starts training.

Run the completed Day-4 implementation smoke with:

    .conda-roborefer/bin/python -m workspace.mh_pcrau_v3.day4_smoke

The Day-5 decision was `G2_PASS` revision 02, which authorized the completed
Day-6 S1a development run. Reasoning/source remain masked without certified
labels; confidence is deferred to OOF/S1b. Calibration, Test and robot
evaluation remain sealed.

Day 6 is now `S1A_COMPLETE`. The best checkpoint and final/optimizer states
are stored under `checkpoints/day06/` and referenced by SHA-256 from
`ketqua1/07_huan_luyen/ngay_06/S1A_CHECKPOINT_MANIFEST.json`. Day 7 OOF/S1b
is authorized; this does not open G3 or any sealed evaluation split.
