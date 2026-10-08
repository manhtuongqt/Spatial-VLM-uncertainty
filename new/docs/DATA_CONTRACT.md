# Archived-data contract

The development manifest is read from `old/protocol` and paths beginning with
the former `datasets/roborefer_dataset_v2_1_1...` root are remapped to the
actual immutable archive directory.  No symlink or edit inside `old/` is
required.

Mandatory preflight invariants:

- exactly 320 train and 80 dev families;
- exactly five variants per family;
- no family overlap;
- 1,600 train and 400 dev samples;
- feature tensors named `R0_GRID`, `D0_GRID`, `R0_THUMB`, `D0_THUMB`;
- feature shape `24×32×1152`, thumbnail shape `1152`;
- every referenced mask/cache file exists;
- optional full SHA-256 verification matches archived manifests.

Supervision is derived as follows:

- target/interior/anchor heatmaps: archived evaluator masks;
- answerability: archived four-state label;
- semantic/relation/depth/occlusion: archived multi-label sources;
- spatial: weak label equal to `AMBIGUOUS`, weighted by 0.5;
- relation-edge validity: positive only when the relation has a visible anchor
  and the archived state is `FOUND`; it is an operational evidence target, not
  a new ground-truth causal claim.

