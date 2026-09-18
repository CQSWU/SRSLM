# Reproducibility

## Fixed behavior

AORePlan calls static-map A* only when the dynamic proposal returns to the
position occupied at the previous timestep. That previous position is updated
on every step, including waits and blocked moves. If static A* has no complete
path, its proposal becomes wait. A static first step into a currently occupied
cell also becomes wait before submission under both `block_both` and `soft`.
The resulting static proposal replaces the dynamic one unless it is still a
reverse move; in that case the dynamic proposal is kept. RePlan's original
no-path fallback remains unchanged. Only the
standalone `AORePlan-SoftNoCheck` search ablation omits this occupancy check;
it is not the planner used inside the retained soft SRSLM result.
Each newly failed dynamic proposal is admitted to the planner's temporary
failure cache with probability 0.5. The draw concerns only the new failure:
older cached destinations stay in place, and a successful move clears the
cache through the native planner feedback path.

ARPE uses a frozen EPOM-L base and the trace branch selected in
`configs/arpe_final_candidate.json`; the historical copy under `artifacts/`
is preserved unchanged. The branch predicts five corrections in
POGEMA action order: wait, up, down, left, and right. The selected checkpoint
is the 500,015,104-frame artifact in
`weights/EPOM-TracePaperConvDirectCorrection-R5-500m/`.

Paper Direct uses the same EPOM-L base and free-cell centering over the whole
11x11 trace crop. When the PRIMAL3 entropy gate opens, it finds the two
statically legal movement directions with the largest original logits and
adds 1 to the lower-pressure one. Wait and all other logits remain unchanged.
The retired signed, clipped-ReLU, candidate-centering and configurable-gate
entrypoints are not part of the reproducible source tree.

SRSLM first obtains complete ARPE and AORePlan candidates. If AORePlan returns
wait, ARPE is used without running Switcher. Otherwise Switcher samples one of
the two branches. The environment and deployed method share this route.

## Retained training configurations

- `learning/train_epom_lifelong_finetune_r5_100m.yaml`
- `learning/train_epom_trace_paper_conv_fusion_r5_500m.yaml`

The retained Switcher is the final new-branch 500M checkpoint:
`weights/SRSLM-switcher-new-branches-500m/SRSLM-Switcher-NewBranches-500M/checkpoint_p0/checkpoint_000122074_500015104.pth`.
Its frozen ARPE candidate is recorded in
`artifacts/retained_srslm_switcher_weights_20260903.json`.

Use `train.py` for ARPE/EPOM training and the dedicated
`train_switcher_wait_arpe.py` or `train_switcher_nowait.py` for Switcher.
The generic training entrypoint rejects Switcher environments. These recipes
are not a promise to recreate a historical checkpoint byte for byte. Keep
the original checkpoint/config hashes and run-bound source for exact artifact
reproduction, and use new output directories and declarations for new training.
DCC-L's selected update-100 fine-tuning code and selection evidence are kept
in the private consolidated package described in `docs/DCC_L_REPRODUCTION.md`.
Use its verified explicit weight path, not the default official DCC path.

## Main exact960 protocol

Every main-table result uses:

- 32 capacity-compatible Moving AI maps;
- 100, 200, 300, 400, 500, and 600 agents;
- seeds 0, 42, 123, 2024, and 3407;
- lifelong `restart` targets;
- separate `block_both` and `soft` execution rules;
- 512 steps and observation radius 5;
- exactly 960 unique, finite, error-free rows per method and rule.

The selected EPOM-L, ARPE, Switcher and DCC-L checkpoints are block-trained
and used unchanged in both execution rules. PRIMAL2 uses its same released
checkpoint in both; CHS uses its audited reconstructed adapter and frozen
EPOM-L. PIBT is no longer a comparison method; its code and results were
removed on 2026-09-10. Follower is kept outside this shared protocol as noted below.

`scripts/validate_paper_exact960.py` and the method-specific validators bind
the protocol, result file, source snapshot, and selected artifacts. Completed
result directories are immutable. Audit the actual run's original source,
checkpoint/config hashes, rows, journal and validation records before paper
use. Current source or newly written documentation cannot repair historical
provenance gaps; retain and report those limitations without editing raw rows
or the historical source manifests.

## SRSLM ablation

The retained hybrid ablation has three rows:

- `SRSLM-OnlyWait`: deterministic wait routing without a learned selector;
- `SRSLM-NoWait`: the learned selector is called on every state;
- `SRSLM`: wait-aware routing plus the learned selector.

The corresponding retained block_both exact960 result identifiers are:

```text
results/srslm_onlywait_final_caar_exact960_20260903
results/srslm_nowait_final_caar_exact960_20260903
results/srslm_wait_aware_caar_100m_exact960_20260903
```

Soft ablations use the same selected weights and conservative occupancy
check as Full SRSLM. Their completed per-arm results must pass the same
audit before being added to a combined table; an in-progress queue or a
COMPLETE marker alone does not establish this. Historical result identifiers
do not imply that every server has every result at its canonical root.

## FoLLow protocol boundary

`agents/follower.py` loads the released official FoLLow checkpoint from
`third_party/learn-to-follow-official/model/follower`. Its current evaluation
uses FoLLow's native `priority` collision protocol. Keep it separate from the
shared `block_both`/`soft` exact960 table and label the protocol explicitly.

## Verification

Run the focused current implementation tests with:

```bash
uv run python -m pytest \
  tests/test_ao_replan.py tests/test_aoreplan_branch.py \
  tests/test_epom_paper_entropy_fusion.py tests/test_switcher.py \
  tests/test_switcher_arpe_candidate.py tests/test_srslm_arpe_ablation.py \
  tests/test_follower_checkpoint_selection.py
```
