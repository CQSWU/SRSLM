# Reproducing the selected DCC-L result

The paper uses the small, early-stopped lifelong fine-tune selected at update
100. It does not use the retired 10,000- or 50,000-update trials. The same
selected checkpoint is evaluated under both `block_both` and `soft`; no
soft-specific model is trained.

## Retained artifacts

- Selected weight: `weights/dcc_l_reward_safe_u100_20260907/dcc_l.pth`.
  SHA256: `e3105b60f0a0b5c57bfafab756b07e55f9902e9cc5d9f275fcd98d8d5cd16915`.
- Official initialization: `otherpolicy/DCC/saved_models/128000.pth`.
  SHA256: `d1a2e848b75c3c94f590004d93bf33365b50ec8e1e011df26c8dc6cc638860f1`.
- Selected results: `results/dcc_l_reward_safe_u100_blockboth_soft_exact960_20260907/`.
- Official comparison results: `results/dcc_official_blockboth_soft_exact960_20260907/`.
- Frozen source and dependency bundle:
  `artifacts/reproducibility/DCC_L_FINAL_EVAL_20260907/frozen_inputs/dcc_l_final_frozen_inputs_20260907.tar.gz`.
  SHA256: `b164533b333879fb30206eab3cfa201f7317d031d87c3345c949e6a3caa8d2e7`.
- Training configuration, selection records and validation evidence:
  `artifacts/reproducibility/DCC_REWARD_PILOT_20260907/` and
  `artifacts/reproducibility/DCC_L_FINAL_EVAL_20260907/`.
- Consolidated readable source and selection evidence:
  `artifacts/reproducibility/DCC_L_SELECTED_COMPLETE_20260908/`.
  This closes the original auditors' 244-file dependency chain, including the
  actual completion markers, selected update100, and update250 safety-stop
  evidence. Run `python verify_consolidated_package.py` from that folder with
  Python and PyYAML to check it without the original server directory layout.
  This read-only check has passed on Windows and both Linux servers.
- Portable archive of that package:
  `artifacts/reproducibility/dcc_selected_complete_v2_20260908.tar.gz`.
  SHA256: `0aab463de8cbb0e66589bee3261329ac83cfc8202a819fbb518befdc5704f7e0`.

Heavyweight artifacts are kept separately from the source repository. A source
checkout alone is not the full reproduction package. The frozen bundle's path
manifest records the original evaluation environment; restore it in an isolated
environment rather than silently substituting the latest development files.
The consolidated package checks selection and development-history evidence;
it does not include the installed PPU runtime or replace the final raw960
results. Its original launch scripts record the two-arm development pilot,
not a new best-only training command. Keep those frozen bytes unchanged.

## Training and selection

The official network and recurrent architecture are unchanged. The retained
training variant uses lifelong restart with `block_both`, a target-completion
reward of +3 and zero other rewards, Adam learning rate 2e-5 and batch size 128.
It retains the inherited DQN workflow and density-based training curriculum.
The replay implementation prevents nonexistent pre-episode history from being
processed as real recurrent input.

The validation check at update 250 triggered a safety stop. Update 100 was
selected because it was the last checkpoint to pass both validation checks,
using four development maps separate from the 32 paper maps. Selection occurred
after the small validation experiment and before the full paper evaluations;
it was not a prespecified update-100 endpoint. It is not a converged 500-update
model. About 2.97 million completed agent-environment steps had been collected
at the selected update; the inherited learner directly supervises focal-agent
TD samples, so this number is not a count of independent learning samples.

## Evaluation

Each execution rule uses 32 capacity-intersection maps, populations
100/200/300/400/500/600, seeds 0/42/123/2024/3407, 512 steps and lifelong
restart. There are 960 unique episodes per rule. The outer observation radius
is 5; DCC retains its native six-channel 9x9 input. Clear its recurrent state
and last-action history between episodes, but retain them when a new goal is
assigned within a lifelong episode.

The retained mean throughputs are 0.19746907552083334 for `block_both` and
1.8524068196614583 for `soft`. The corresponding official-checkpoint means are
0.191064453125 and 1.8011088053385416. These are small average improvements,
not improvements at every population and not a claim of statistical
significance. Per-rule validation manifests and the independent paired audit
bind the original rows, journal entries, checkpoints and source hashes.

## Cleanup boundary

Obsolete 50k-run checkpoints, failed trial outputs and obsolete update-10k
entry points have been retired. Do not recreate their old paths or restart
those training schedules. The official initialization, selected checkpoint,
final comparisons and source/selection evidence remain necessary for paper
reproduction. Cleanup receipts are kept with the local reproducibility
artifacts; they are not experiment results.
Unselected small-pilot weights, old resume payloads and unused update200
weights have also been retired. Update250 is retained solely because the
selection auditor verifies it as evidence of the safety stop.
