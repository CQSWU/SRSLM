# SRSLM current version

This repository contains one retained paper implementation, documented on 2026-09-08.
Exploratory variants from earlier iterations are not part of this snapshot.
For the verified September 10 cleanup closeout and remaining limitations, see
`docs/CLEANUP_CLOSEOUT_20260910.md`.

## AORePlan accumulated observed static map (2026-09-12)

Static A* now retains each agent's own observed static obstacles across all
steps of an episode, including steps without a probe, skipped planning and
at-goal steps. Target changes do not erase this map. Episode reset creates new
private static planners. No full environment map or other agents' map memory
is accessed; never-observed cells keep the original free-cell assumption.
The static query ignores agents and failed-action memory, while its first
step still receives the existing current-observation occupancy check. The
explicit standalone `AORePlan-SoftNoCheck` ablation still omits that check.

The shared implementation also changes future SRSLM planner queries. Dynamic
planning, reverse detection, cache release and no-path fallback are unchanged.
The paper's failed-proposal rule is now part of the production planner: a newly
failed dynamic proposal enters the temporary failure cache with probability
0.5. Dropping a new failure does not erase older cached failures, and successful
movement keeps the native cache-clearing behavior. Completed result directories
remain immutable evidence and must retain their recorded source hashes.

## ARPE training trace auto-reset fix (2026-09-12)

`TauObservationWrapper` now detects when the inner POGEMA environment has
created a new grid. It clears the previous episode's trace, binds the new
obstacle map, and deposits the new initial occupancy exactly once. This also
handles automatic resets that do not call the outer wrapper's `reset()`.
A terminal frame without auto-reset and a lifelong target reassignment do
not clear the trace. This detection relies on the pinned POGEMA reset contract
that each episode creates a new `Grid` or `GridLifeLong` object.

Only training observation lifecycle handling changed. Network structure,
selected weights, Switcher routing and historical evaluation data are
unchanged. Existing weights are not retroactively repaired by this source
fix; any training comparison must have its own provenance. Regression and
deployment evidence is in `results/trace_autoreset_fix_20260912/`.

## AORePlan cache release fix (2026-09-12)

The active source now releases temporary failed destinations when an actual
dynamic planning query, including BestMove, returns no action. That step keeps
the original random-or-stay fallback. The next observed planning step can retry
without stale failure constraints; real obstacles and observed agents remain
blocked. Skipped queries and cancelled valid proposals do not release this
cache. RePlan, reverse detection, static-step checking and fallback sampling
are otherwise unchanged.

The matched 18-episode trajectory check is documented in
`results/aoreplan_cache_release_fix_20260912/cache_fix_comparison.md`.
This is not a new full-paper evaluation. The shared AORePlan implementation
also affects future SRSLM executions; historical AORePlan and SRSLM results
retain their original source hashes and must not be relabelled as fixed runs.
On Linux, deploy the matching native planner binary or rebuild `planner.cpp`
before using the new Python caller. Windows uses the updated Python fallback.

## Retired branches (2026-09-10)

PIBT, the learned Follower Trace branch, and SRSLMF are retired from the
active source tree. Retention of historical artifacts is selective; consult the
cleanup records rather than assuming all retired results and weights remain.
The official Follower and its parameter-free
Direct variants, with and without entropy gating, remain supported by the
retained isolated Direct entry point. The paper SRSLM and ARPE implementations
are unchanged. Historical launch notes are under `docs/archive/` and are not
instructions to restart retired experiments.

## Current soft execution contract

The retained paper implementation uses conservative static-step occupancy
checking in both `block_both` and `soft`, including inside SRSLM and its
switching ablations. An occupied static-A* destination becomes wait before
submission. The same block-trained selected weights are used under soft.

The standalone soft search ablation is the explicit
`AORePlan-SoftNoCheck` method. It submits that static action without the
extra occupancy check. This exception does not change SRSLM. Earlier frozen
packages/results retain their original contracts and hashes; they must not
be relabelled as the new canonical source.

The internal runner retains external comparison methods as well as the
publication's current self-owned method bindings. Its public Git counterpart
intentionally has a smaller registry. No experiment result is changed by this
source-only synchronization.

## Retained methods

- Planning: `RePlan` and `AORePlan`.
- Learned policies and ablations: `EPOM`, `EPOM-Lifelong-FT`, `Direct`, and
  `ARPE`.
- Hybrid method and ablations: `SRSLM`, `SRSLM-NoWait`, and
  `SRSLM-OnlyWait`.
- Retained comparisons: `DCC` (selected DCC-L with an explicit weight path), `PRIMAL2`, `Follower`, and `CHS-Reconstructed`
  together with the EPOM-L variants explicitly listed by the runner.

`Follower` is the official Learn-to-Follow policy. It remains opt-in because
its released configuration uses its native `priority` collision protocol rather
than the `block_both` and `soft` execution rules reported in the main table.
CHS uses the audited reconstructed adapter and frozen EPOM-L, not a newly
trained CHS model. Its selected soft-compatible adapter accepts both main-table
execution rules.

The AS comparison, the NoReweight runner entry and the historical
NoReweight-based Direct were retired on 2026-09-10 because the final paper
does not use them. The remaining NoReweight encoder, configuration, CLI field,
training recipe and no-entropy ARPE candidate were removed on 2026-09-14.
`PolicyBackbone` remains because the selected ARPE inference adapter uses it.

## Current implementation

- `planning/replan_algo.py` and `planning/ao_replan_algo.py` implement the two
  search methods. AORePlan checks a reverse proposal with static-map A*.
- `agents/epom_trace_context.py` and
  `learning/epom_trace_multiplier_actor_critic.py` implement the selected
  ARPE trace-reweighting branch on the frozen EPOM-L base.
- `agents/switcher.py`, `agents/switcher_core.py`, and
  `learning/switcher_actor_critic.py` implement the retained two-branch
  Switcher. An AORePlan wait directly selects ARPE; otherwise the learned
  Switcher samples between ARPE and AORePlan.
- `agents/srslm.py` is the deployed composition. Training and inference share
  the same routing logic.
- `agents/follower.py` and `third_party/learn-to-follow-official/` contain the
  retained official FoLLow adapter and pinned release.
- `run_experiments.py` is the canonical internal evaluator, including external method entries.

## Selected artifacts

- ARPE: `weights/EPOM-TracePaperConvDirectCorrection-R5-500m/EPOM-TracePaperConvDirectCorrection-R5-S0-20260902`,
  selected checkpoint `checkpoint_000122074_500015104.pth`.
- Final new-branch Switcher:
  `weights/SRSLM-switcher-new-branches-500m/SRSLM-Switcher-NewBranches-500M`,
  checkpoint `checkpoint_000122074_500015104.pth`.
- Its frozen branch-zero candidate is
  `weights/trace_residual_shaped200/TraceBonusSync200Entropy-20M-S0`,
  checkpoint `best_000057500_235520000_avg_throughput_1.640.pth`.
- The selected ARPE and frozen EPOM-L identities are pinned by
  `artifacts/arpe_final_candidate.json`.

The retained exact960 SRSLM results are:

- `results/srslm_wait_aware_caar_100m_exact960_20260903`
- `results/srslm_nowait_final_caar_exact960_20260903`
- `results/srslm_onlywait_final_caar_exact960_20260903`

These are historical result identifiers, not a claim that every server has
every result directory at its canonical root. Formal result directories and
checkpoint hashes are immutable evidence. New runs must use a new output
directory; deployment of current source does not change a completed run.

## Current artifact declarations

`configs/arpe_final_candidate.json` is the path-relative declaration for the
retained gated ARPE; the historical `artifacts/caar_final_candidate.json`
remains untouched as immutable historical evidence. New training must use
separate output directories and new declarations rather than reuse historical
hashes.

Direct explicitly uses EPOM-L and whole-crop free-cell centering. When the
PRIMAL3 entropy gate opens, it takes the two statically legal movement
directions with the largest original logits and adds 1 to the lower-pressure
one. Wait and all other logits are unchanged. The retired signed, clipped-ReLU,
candidate-centering and gate-selection CLI paths have been removed. DCC-L's selected
checkpoint and provenance are documented in `docs/DCC_L_REPRODUCTION.md`;
its code/weights remain private third-party dependencies, not a public source
release. Selected paper result rows and pinned checkpoint files are unchanged;
unneeded artifacts have been retired as documented in the cleanup records.

## Server artifact availability

Source, installed runtime, weights and result evidence are separate parts of
reproduction. The following locations were checked on 2026-09-08; check their
current hashes again before launching anything. Source synchronization does
not install packages, move weights or change frozen experiments.

- The gated ARPE, EPOM-L, Full and NoWait artifacts declared above exist at
  the canonical relative paths on both servers. OnlyWait uses the same ARPE
  and has no separately trained Switcher checkpoint.
- DCC-L's portable layout `weights/dcc_l_reward_safe_u100_20260907/dcc_l.pth`
  is installed at both S1 and S2 canonical roots. The selected checkpoint also remains available on
  both servers at the following path relative to that root:
  `artifacts/reproducibility/DCC_L_SELECTED_COMPLETE_20260908/private/frozen/root/dcc-reward-pilot-20260907/weights/throughput/100.pth`.
  Its SHA256 is
  `e3105b60f0a0b5c57bfafab756b07e55f9902e9cc5d9f275fcd98d8d5cd16915`.
  Pass its resolved absolute path to `--dcc-weights-path`. S1 also retains
  the original `/root/dcc-reward-pilot-20260907/weights/throughput/100.pth`.
  Do not fall back to the official checkpoint and label it DCC-L.
- Official FoLLow loads from
  `third_party/learn-to-follow-official/model/follower`, not `weights/Follower`.
  Its selected release is present on both servers and remains priority-only
  for the retained comparison protocol.
- PRIMAL2's converted checkpoint and planner source are present on both
  servers.

The consolidated DCC-L evidence package is private, hash-verified research
material. Neither it nor third-party source/checkpoints is included in the
public GitHub source release.
