# Reproducibility

This note covers the small public method registry. Checkpoints and experiment
evidence are distributed separately from Git. Artifact identities and the
retained numerical result are listed in [CURRENT_VERSION.md](../CURRENT_VERSION.md);
fresh-run examples are in [README.md](../README.md).

## Execution rules

Run each method separately under `block_both` and `soft` on the same grid:
32 held-out capacity-intersection maps, populations 100/200/300/400/500/600,
seeds 0/42/123/2024/3407, lifelong `restart`, 512 steps and radius 5. This is
960 unique map-population-seed episodes per method and rule. Keep the same
selected block-trained EPOM-L, CAAR and Switcher weights in both rules.

AORePlan first obtains the dynamic planner's proposal, including BestMove.
If there is no proposal, it uses RePlan's original random-or-stay fallback
without a static A* query. Otherwise, a proposal that returns to the position
at the previous timestep is reverse and triggers static-map A*. The previous
position is recorded even after waits or blocked moves. If static A* returns
no action, use wait. If its first step targets a currently occupied cell,
also use wait. After these guards, keep the dynamic proposal only when the
static proposal is still reverse; otherwise replace it, including with wait.
This conservative occupancy check also applies under soft inside SRSLM and
its switching ablations.

The only exception is the separately named `AORePlan-SoftNoCheck` standalone
search ablation. Under soft it omits the extra static-step occupancy check;
it does not change the planner inside the retained SRSLM result.

## Learned and fixed reweighting

Paper Direct uses frozen EPOM-L, signed pressure and free-cell centering over
the entire 11x11 trace crop. Entropy gating and clipped ReLU are separate,
explicitly recorded options. Clipped ReLU applies after centering. Learned
CAAR is not changed by these Direct options.

The gated CAAR declaration is `configs/caar_final_candidate.json`. The
independently trained 500M-step ungated CAAR uses
`configs/caar_noentropy_candidate.json`; it is not the gated checkpoint with
its gate disabled only at inference. Both use the same frozen EPOM-L base.
The historical NoReweight backbone is a different model, not an EPOM-L alias.

Full SRSLM, NoWait and OnlyWait share the same selected gated CAAR. NoWait
uses its independently trained all-state Switcher. OnlyWait has no Switcher
checkpoint. Their soft runs keep Full's conservative occupancy check.

Declarations identify required artifacts; they do not supply the files.
Verify the checkpoint and config hashes before inference. New training needs
new run directories, artifact declarations and hashes. A recorded recipe
does not promise byte-identical regeneration of a historical checkpoint.

## Result evidence

For each result, check the exact grid and execution rule, all 960 unique
finite error-free rows, selected checkpoint/config hashes and method-specific
episode reset behavior. Keep the original source snapshot and validation
records bound to the raw result. Where an original journal exists, compare
every episode with the reported rows. Do not accept an incomplete soft arm
as a completed ablation or infer correctness from a COMPLETE marker alone.

Historical raw rows, source hashes and limitations stay unchanged. A new
validator, documentation update or current source tree cannot establish
missing historical evidence retroactively. Explicitly report unavailable
source or journal records instead of presenting reconstructed records as
original evidence. Current curated source and an original frozen run may
have different hashes; identify which one an experiment actually used.

This public source release does not include external comparison adapters,
third-party checkpoints or private experiment bundles. Their original
licenses and separate reproducibility requirements remain applicable; they
are not brought under this project's license by a documentation update.
