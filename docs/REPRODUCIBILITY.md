# Reproducibility

This note covers the public method registry. Checkpoints and complete experiment
evidence are distributed separately from Git. Selected identities are listed in
[`CURRENT_VERSION.md`](../CURRENT_VERSION.md).

## Fixed method behavior

AORePlan first obtains RePlan's dynamic proposal, including its original
BestMove/no-path fallback. A proposal returning to the previous timestep's
position triggers A* on the accumulated observed static map. The previous
position is updated every step, including waits and blocked moves. If the
static query has no first action or that action targets a currently occupied
cell, it becomes wait. A successful move releases the relevant planner failure
cache. New failed destinations use the retained randomized caching rule.

The separately named `AORePlan-SoftNoCheck` search ablation omits the local
occupancy check under `soft`. It is not the planner used by SRSLM.

Direct and ARPE share the frozen EPOM-L base and whole-crop, free-cell-centred
11x11 shared trace. Direct has no trainable trace parameters. ARPE trains only
its trace branch and independent critic while keeping EPOM-L frozen. Trace,
grid memory, recurrent state, and planner state are cleared through their
method-specific episode reset paths.

SRSLM first obtains complete AORePlan and ARPE proposals. An AORePlan wait uses
ARPE without invoking Switcher. Every other state is sent to the two-action
Switcher. Training and inference call the same routing implementation.

## Evaluation grid

The current paper evaluation contains 36 maps:

- `maps/eval_capacity_intersection_n600.yaml` (32 maps);
- `maps/eval_wc3_extra3.yaml` (three resized MovingAI WC3 maps);
- `maps/eval_wc3_extra1_timbermawhold.yaml` (one resized MovingAI WC3 map).

For each method and execution rule, run populations
100/200/300/400/500/600 and seeds 0/42/123/2024/3407 with lifelong `restart`,
512 steps, and radius 5. The complete grid therefore has 1,080 unique rows.
Evaluate `block_both` and `soft` separately with the same selected learned
weights. The older 32-map bundles contain 960 rows and remain separately named
historical evidence.

## Result audit

Before using a result, require:

1. the exact map/population/seed grid and execution rule;
2. unique, finite, error-free rows;
3. the original map, source, configuration, and checkpoint hashes;
4. method-specific reset checks for recurrent, trace, grid-memory, and planner
   state;
5. the run's original journal and validation record when available.

A completion marker alone is not validation. Current source or newly written
documentation cannot retroactively fill a missing historical source snapshot.
Keep each completed output directory immutable and report evidence gaps rather
than reconstructing provenance as if it were original.

The public release does not include external comparison adapters, third-party
checkpoints, or private experiment bundles. Their licenses and reproduction
requirements remain separate.
