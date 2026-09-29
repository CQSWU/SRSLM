# Current implementation

Updated 2026-09-29. Current SRSLM uses only the wait rule. Previously completed
two-rule experiments are recorded separately below; their results are unchanged.

2026-09-30: removed mandatory paper-audit checks and fixed training-recipe
restrictions from public entry points. Recorded hashes remain informational.
Default weights, inference settings and the wait-only rule are unchanged.

The follow-up cleanup shares Switcher input definitions, deployment hooks and
worker population assignment, and removes unused audit helpers. Checkpoint
tensors, action sampling and algorithm defaults are unchanged.

Public evaluation now exposes only AORePlan, ARPE and SRSLM. Standalone
RePlan/EPOM-L, Direct, switching ablations and zero-trace control code have
been removed. Earlier sources, results and weights remain in the offline
archive. The required EPOM-L backbone and ARPE training bonus are retained.

## Selected inference settings

- AORePlan uses the accumulated observed static map, reverse detection, local
  occupancy checking, cache release, and randomized failure caching with
  probability 0.5.
- ARPE uses frozen EPOM-L and the 1B-step learned Trace branch. If the base
  policy's Shannon entropy exceeds 0.01, add twelve times the learned residual;
  otherwise use the base logits unchanged. The residual is `0.5*tanh(raw)`
  minus its five-action mean. Direct's fixed bonus is disabled in this final
  inference configuration.
- Full SRSLM uses ARPE directly when AORePlan proposes wait. Otherwise,
  including a reverse move, the frozen Switcher selects between the proposals.

The ARPE checkpoint was trained with the always-on learned correction and
Direct bonus 1. The final entropy threshold and multiplier are inference
settings. The Switcher was trained from scratch for 1B steps with alternating
populations and the earlier 500M ARPE candidate, then evaluated with the above
1B ARPE candidate. It was not retrained for that final candidate replacement.
Saved training configurations are retained separately from inference settings.
This wait-only code update does not retrain or replace any checkpoint. New
Switcher training and inference share the same wait-only controller.

## Checkpoints

Place the separately supplied checkpoint folders in `weights/`:

| Folder | SHA256 of selected checkpoint |
| --- | --- |
| `EPOM-L` | `f70a305ee68546be95e0a93d7f61c9aec435a50da20624a3b382af2276ad79d2` |
| `ARPE-Final-1B` | `4c57a46369fe7e450b74346a0422e74709d32ffe4ee6179fdc02bc8e1346a9c6` |
| `SRSLM-Switcher-Final-1B` | `5632b18c2902311aef8cd6e5467ab29cb5196eda35e0759e405ea9e90ca55144` |

Hashes identify recorded artifacts; they are not runtime allowlists. Same-named
older checkpoints are not interchangeable. Relative paths and inference
settings are in `configs/arpe_final_candidate.json`.

## Historical evaluation evidence

The SRSLM and OnlyRule results below used wait and reverse overrides before
the wait-only update. They are not performance measurements of current SRSLM.

The full grid is 36 maps in `maps/test.yaml`, populations 100/200/300/400/500/600,
and seeds 0/42/123/2024/3407: 1,080 episodes per method and execution rule.
Episodes use lifelong restart, 512 steps, and observation radius 5. The same
weights are evaluated separately with `block_both` and the original `soft`
execution implementation. The 4096-step experiments use their separate
eight-map, two-population, three-seed contracts (48 episodes).

| Result | Mean throughput |
| --- | ---: |
| Full SRSLM, block_both | 2.4024703414 |
| Full SRSLM, soft | 3.1037995515 |
| NoRule, block_both | 2.3436234086 |
| OnlyRule, block_both | 2.0812952112 |
| ARPE, block_both | 2.2882233796 |
| ARPE, always x12 | 2.0126989294 |
| ARPE, zero Trace input | 1.6863516348 |

Raw rows, contracts, source snapshots and validation manifests are kept with
the separate data bundle. Source code alone is not that data bundle. Do not
replace historical run identities or label two-rule results as wait-only.

## Release scope

Git contains portable method code, tests and map splits, not checkpoints, raw
results, environments or third-party comparison implementations. Exact as-run
sources are preserved before cleanup. External comparisons retain their own
licenses, checkpoints and provenance.
