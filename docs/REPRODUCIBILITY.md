# Reproducibility

This note covers the public method registry. Checkpoints and complete experiment
evidence are distributed separately from Git. The loaders accept ordinary
relative or absolute weight paths and do not enforce paper checkpoint hashes.
Selected paper identities are listed in [`CURRENT_VERSION.md`](../CURRENT_VERSION.md)
only for provenance.

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

The selected ARPE checkpoint computes `z + g*Direct + g*residual`, where
`residual = 0.5*tanh(raw) - mean(0.5*tanh(raw))` across the five actions.
Direct adds one to the lower-pressure direction among the two highest-logit
legal moves, when at least two moves are legal. The residual can affect all
five actions and is not action-masked. Actor and critic have separate trace
encoders and fusion layers. Checkpoint version buffers and every model tensor
must match; a same-shaped actor head alone does not establish compatibility.
Standalone ARPE evaluation retains the historical Direct-style NumPy sampler;
the SRSLM learning branch retains PyTorch sampling as used during Switcher
training. Both paths record their sampler and reset its state per episode.
The ARPE training recipe also retains the individual failed-move credit:
an attempted move that does not change position receives an extra -0.0098
on top of the base -0.0002 failed-move penalty. Explicit wait is not charged
this extra penalty. No team reward is added. This shaping affects training
rewards only; it does not change evaluation throughput or collision rules.

SRSLM first obtains complete AORePlan and ARPE proposals. An AORePlan wait uses
ARPE without invoking Switcher. Every other state is sent to the two-action
Switcher. Training and inference call the same routing implementation.

## Evaluation grid

The current paper evaluation uses `maps/test.yaml`. It contains all 36 maps:
the fixed 32-map capacity-intersection set and four resized MovingAI WC3 maps.
The grids are stored directly in this file, so no secondary registry is needed.

For each method and execution rule, run populations
100/200/300/400/500/600 and seeds 0/42/123/2024/3407 with lifelong `restart`,
512 steps, and radius 5. The complete grid therefore has 1,080 unique rows.
Evaluate `block_both` and `soft` separately with the same selected learned
weights. The older 32-map bundles contain 960 rows and remain separately named
historical evidence.

## Result audit

For a paper-quality audit, record:

1. the exact map/population/seed grid and execution rule;
2. unique, finite, error-free rows;
3. the original map, source, configuration, and checkpoint hashes;
4. method-specific reset checks for recurrent, trace, grid-memory, and planner
   state;
5. the run's original journal and validation record when available.

These audit records are recommended for publication but are not runtime gates.
Replacing or retraining a checkpoint therefore does not require editing a hash
allowlist in the source code.

Result journals record a source fingerprint automatically. A journal from an
older implementation cannot be resumed under changed source, even if its manual
protocol label is reused. Start a new result directory after changing the
implementation. Do not edit an evaluation's source tree while it is running.

Only the retained entropy-gated ARPE equation is loadable. Inference overrides
for removing the gate or changing its threshold, automatic best/latest fallback,
the shuffled-trace branch, and the unused team-reward wrapper have been removed.
The real-trace and zero-trace training recipes remain. A zero-trace result is a
valid matched control only if it was trained using this same model and budget.
The explicitly declared base checkpoint must match the one actually loaded;
finding another checkpoint in the same directory does not make it equivalent.

Run `python -m pytest -q` before deployment. The same regression suite runs on
GitHub pushes and pull requests. This checks code contracts, not the validity of
historical experimental results or any unprovided private checkpoint.

The public release does not include external comparison adapters, third-party
checkpoints, or private experiment bundles. Their licenses and reproduction
requirements remain separate.
