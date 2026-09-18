# Current public implementation

Updated 2026-09-18 from the active Server 1 source tree, followed by the final
public-source cleanup. This repository intentionally contains only self-owned
method code, focused tests, and portable evaluation utilities. Weights, raw
results, server environments, private adapters, and third-party repositories
are outside the Git release.

## Retained method path

- AORePlan uses the accumulated observed static map, current reverse detection,
  conservative local occupancy check, cache-release fix, and randomized failure
  caching.
- ARPE uses the selected `paper_entropy_fusion` trace branch on a frozen EPOM-L
  base. Trace state is cleared at every episode boundary.
- SRSLM uses ARPE immediately for an AORePlan wait; otherwise Switcher samples
  between the complete ARPE and AORePlan proposals.
- NoWait and OnlyWait remain the two switching ablations. Retired historical
  branches are not aliases and cannot be selected through the public runner.

## Selected artifact identities

Weights are not committed. The following hashes identify the selected paper
artifacts and must be verified before inference:

- EPOM-L base checkpoint:
  `f70a305ee68546be95e0a93d7f61c9aec435a50da20624a3b382af2276ad79d2`
- ARPE 500M selected checkpoint:
  `497118e3aa4fbaecde35e53f31fe3126e11c1a1e5b0b621b89ac0d340002d41b`
- final 1B Switcher checkpoint:
  `65c255ad9a3ae0c5874001637f4a4ff4d8172bd5b0b2e9b5d5a8d0c156d9e547`
- final 1B Switcher saved configuration:
  `07ce3ef6d2e57dea46760cf5b98363cd1f11d6ecfad84e8f1fe707c198e12d52`

`configs/arpe_final_candidate.json` is the path-relative ARPE declaration. A
declaration records an identity; it does not supply or download the files.

## Current paper grid

The current comparison uses 36 maps, six populations, and five seeds, for 1,080
episodes per method and execution rule:

- maps: the 32-map capacity-intersection registry plus four resized MovingAI
  WC3 maps;
- populations: 100, 200, 300, 400, 500, and 600;
- seeds: 0, 42, 123, 2024, and 3407;
- lifelong target replacement, 512 steps, observation radius 5;
- separate `block_both` and `soft` evaluations using the same selected
  block-trained learned weights.

Older exact960 result names identify the original 32-map experiments. They
remain valid historical evidence but are not the current 36-map aggregate.

## Public/private boundary

The public runner exposes RePlan, AORePlan, AORePlan-SoftNoCheck, EPOM-L,
Direct, ARPE, SRSLM-NoWait, SRSLM-OnlyWait, and SRSLM. External comparison
adapters and separately licensed sources/checkpoints are deliberately not
vendored. Generated binaries, caches, logs, results, and weights are ignored.
