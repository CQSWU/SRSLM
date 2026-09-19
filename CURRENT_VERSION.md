# Current public implementation

Updated 2026-09-19 from the active Server 1 source tree, followed by the final
public-source cleanup and layout consolidation. This repository intentionally
contains only self-owned method code, focused tests, and portable evaluation
utilities. Weights, raw results, server environments, private adapters, and
third-party repositories are outside the Git release.

## Retained method path

- AORePlan uses the accumulated observed static map, current reverse detection,
  conservative local occupancy check, cache-release fix, and randomized failure
  caching. When a submitted dynamic-planner proposal fails to move, a separate
  seeded coin admits that destination to the failure cache with probability
  0.5. This admission rule is separate from both exhausted-cache release and
  the original random no-path fallback.
- ARPE uses the selected `paper_entropy_fusion` trace branch on a frozen EPOM-L
  base. The selected checkpoint uses Direct's entropy-gated bonus plus five
  learned residuals: `0.5*tanh(raw)`, then subtract their five-action mean.
  The learned residual uses the same entropy gate. Actor and independent
  critic have 605,638 trainable parameters in total. Trace state is cleared
  at every episode boundary.
- SRSLM uses ARPE immediately for an AORePlan wait; otherwise Switcher samples
  between the complete ARPE and AORePlan proposals.
- NoWait and OnlyWait remain the two switching ablations. Retired historical
  branches are not aliases and cannot be selected through the public runner.

## Selected artifact identities

Weights are not committed. The following hashes identify the selected paper
artifacts. Loading checks the network and forward-rule version, not a hash allowlist:

- EPOM-L base checkpoint:
  `f70a305ee68546be95e0a93d7f61c9aec435a50da20624a3b382af2276ad79d2`
- ARPE selected checkpoint at 235.52M frames:
  `1da454620520a9095a1140cccd8c1829c0fe74063b6f0e405c2c176b0890c9f6`
- ARPE saved configuration:
  `c1f2a9071df428b377e2803bd78913bfa7dd30648a6e1a16f315c5cacb1c190f`
- final 1B Switcher checkpoint:
  `65c255ad9a3ae0c5874001637f4a4ff4d8172bd5b0b2e9b5d5a8d0c156d9e547`
- final 1B Switcher saved configuration:
  `07ce3ef6d2e57dea46760cf5b98363cd1f11d6ecfad84e8f1fe707c198e12d52`

`configs/arpe_final_candidate.json` is the path-relative ARPE declaration. A
declaration records an identity; it does not supply or download the files.

## Current paper grid

The current comparison uses 36 maps, six populations, and five seeds, for 1,080
episodes per method and execution rule:

- maps: `maps/test.yaml`, containing the 32-map capacity-intersection set plus
  four resized MovingAI WC3 maps;
- populations: 100, 200, 300, 400, 500, and 600;
- seeds: 0, 42, 123, 2024, and 3407;
- lifelong target replacement, 512 steps, observation radius 5;
- separate `block_both` and `soft` evaluations using the same selected
  block-trained learned weights.

Older exact960 result names identify the original 32-map experiments. They
are historical evidence, not the current 36-map aggregate. Always retain each
run's original source and checkpoint provenance.

## 2026-09-19 checkpoint compatibility correction

The selected ARPE checkpoint declares `allaction_residual_version=2`. A previous
public runtime incorrectly interpreted its raw head output as an unbounded
subtractive correction and omitted Direct. The model and strict checkpoint
loader now enforce the trained Direct-plus-bounded-residual computation and
restore the independent trace critic. Stored tie rankings keep Direct routing
identical during rollout and PPO replay. Mismatched old checkpoints are rejected
instead of silently skipping their critic and version tensors.

Evaluations made with the mismatched runtime must not be described as results
of the selected trained policy. This also invalidates the attempted zero-trace
control trained with that different architecture; its artifacts are retained
for diagnosis, not a capacity-matched comparison. The zero-trace configuration
now uses the corrected architecture, but that control still requires a new run.

## Public/private boundary

The cleanup following the checkpoint correction removes obsolete inference
gate overrides, auto-selection fallback, shuffled trace, and team-reward code.
The selected forward equation and checkpoint tensor names are unchanged.
Base-artifact identity is checked against the files actually loaded. Journal
resume is source-bound; old result files are never automatically relabelled as
corrected evaluations. GitHub CI runs the regression suite on every code change.

The public runner exposes RePlan, AORePlan, AORePlan-SoftNoCheck, EPOM-L,
Direct, ARPE, SRSLM-NoWait, SRSLM-OnlyWait, and SRSLM. External comparison
adapters and separately licensed sources/checkpoints are deliberately not
vendored. Generated binaries, caches, logs, results, and weights are ignored.
