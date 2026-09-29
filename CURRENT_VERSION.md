# Current implementation

Public methods: **AORePlan, ARPE and SRSLM**.

- AORePlan uses reverse detection, accumulated observed static A*, local
  occupancy checking, exhausted-cache release and randomized failure caching
  with probability 0.5.
- ARPE freezes EPOM-L and uses the 1B-step Trace branch. When the base policy's
  Shannon entropy exceeds 0.01, it adds twelve times the learned residual;
  otherwise it leaves the base logits unchanged. The residual is
  `0.5*tanh(raw)` minus its five-action mean. Direct's bonus is off at inference.
- SRSLM uses ARPE when AORePlan proposes wait. For every non-wait proposal,
  including reverse moves, the Switcher samples between the two branches.

## Weights and training

Place the supplied `EPOM-L`, `ARPE-Final-1B` and `SRSLM-Switcher-Final-1B`
folders in `weights/`, retaining their `config.json` and checkpoint files.
ARPE paths and inference settings are in `configs/arpe_final_candidate.json`.

The ARPE checkpoint was trained with an always-on learned correction and
Direct bonus 1. The final gate and multiplier are inference settings. The
Switcher was trained from scratch for 1B steps with alternating populations
and the earlier 500M ARPE candidate, then evaluated with the 1B ARPE candidate.
It was not retrained for that replacement or this wait-only code update.
New Switcher training and inference use the same wait-only controller.

Weights, results and original training records stay in the offline bundle.
Historical SRSLM results used wait and reverse overrides; they must not be
relabeled as measurements of the current wait-only implementation.
Baseline and ablation entries are not part of this public source tree.
