# Architecture contract

**Opt-in spatial variant (2026-10-05):** the implemented live RGB-D →
phrase-conditioned anchor → horizontal verifier → calibrated risk → perception
decision path is documented in [UNIFIED_SPATIAL_INFERENCE.md](UNIFIED_SPATIAL_INFERENCE.md).
It has a separate locked bundle and does not replace the selected baseline below.

`PCRAUTargetV2` is a sidecar; RoboRefer remains frozen.  The model consumes
cached pre-projector RGB/depth grids (`24×32×1152`) and prompt text.

1. A Transformer contextualizes hashed prompt tokens.
2. Learned target, relation and anchor slots cross-attend to those tokens.
3. RGB and depth projections are fused by a relation-conditioned gate and
   cross-modal residual blocks.
4. Slot-conditioned spatial heads produce target, interior and anchor logits.
5. Soft spatial moments and node features form a predicted query graph.
6. Relation-edge, answerability and five-source heads operate on deployable
   predicted evidence only.
7. Post-processing extracts MAP/top-k modes/covariance.
8. A separately fitted calibrator maps observable evidence to `r_ground`.

Oracle masks and labels enter only `compute_losses` and evaluation.  They are
never members of `model.forward`'s accepted batch keys.

The model supports three relation/anchor slots as required by the target
contract.  The archived development set uses at most two anchors and one
relation, so unused capacity is masked and cannot be claimed as empirically
validated multi-clause reasoning.
