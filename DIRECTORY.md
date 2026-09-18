# Project layout

PIBT, learned Follower Trace, and SRSLMF active implementations were removed
on 2026-09-10. Historical evidence is retained selectively as recorded in
`docs/CLEANUP_CLOSEOUT_20260910.md`. The standalone Follower Direct files in the local development
tree are retained; no learned Follower trainer or SRSLMF launcher remains.

| Path | Purpose |
| --- | --- |
| `agents/replan.py` | RePlan adapter. |
| `agents/ao_replan.py` | AORePlan adapter. |
| `planning/replan_algo.py` | Dynamic replanning and the original no-path fallback. |
| `planning/ao_replan_algo.py` | Reverse detection and static-A* replacement. |
| `planning/aoreplan_branch.py` | Proposal/commit interface shared with SRSLM. |
| `agents/epom.py` | Official EPOM and EPOM-L inference. |
| `agents/epom_trace.py` | Shared trace state used by ARPE. |
| `agents/epom_trace_context.py` | Selected ARPE inference adapter. |
| `learning/epom_trace_context_actor_critic.py` | Frozen EPOM-L base integration. |
| `learning/epom_trace_multiplier_actor_critic.py` | Selected trace encoder, actor, and critic. |
| `agents/policy_backbone.py` | Shared checkpoint loading and trace-state support for ARPE. |
| `agents/epom_direct_reweight.py` | Current EPOM-L Direct with explicit gate/centering/transform. |
| `agents/ao_replan_soft_ablation.py` | Standalone soft search exception, not used by SRSLM. |
| `agents/dcc.py` | Private DCC/DCC-L inference adapter; needs selected external source. |
| `otherpolicy/DCC/model.py`, `otherpolicy/DCC/config.py` | Private selected DCC network and configuration; preserve frozen-source aliases. |
| `agents/chs.py`, `planning/dstar_lite.py` | Audited CHS reconstruction with the frozen EPOM-L base. |
| `agents/primal2.py` | Released PRIMAL2 adapter using the separately stored converted checkpoint. |
| `agents/switcher.py` | Wait-aware Switcher inference. |
| `agents/switcher_core.py` | Shared routing and feature construction. |
| `learning/switcher_actor_critic.py` | Two-action Switcher actor-critic. |
| `pomapf_env/switcher_env.py` | Shared reset/step/reward helpers; not a direct legacy training entry. |
| `pomapf_env/switcher_arpe_env.py` | Current pinned-candidate Switcher training environment. |
| `agents/srslm.py` | Full SRSLM method. |
| `agents/srslm_arpe_ablation.py` | NoWait and OnlyWait ablations. |
| `agents/follower.py` | Official FoLLow inference adapter. |
| `third_party/learn-to-follow-official/` | Pinned official FoLLow source and checkpoint. |
| `learning/*.yaml` | Retained EPOM-L, ARPE and Switcher formal/smoke recipes. |
| `configs/arpe_final_candidate.json` | Hash-pinned selected ARPE candidate. |
| `docs/DCC_L_REPRODUCTION.md` | Selected update-100 lifelong fine-tuning protocol and evidence; portable paths are not installed automatically. |
| `artifacts/reproducibility/DCC_L_SELECTED_COMPLETE_20260908/` | Private consolidated DCC-L selection/source evidence and preserved selected weight. |
| `run_experiments.py` | Internal evaluator retaining current external comparison entries. |
| `train.py` | Shared CAAR/EPOM training entry point. |
| `train_switcher_wait_arpe.py` | Wait-aware Switcher training entry point. |
| `train_switcher_nowait.py` | NoWait ablation training entry point. |
| `maps/` | Training and evaluation map registries. |
| `weights/` | Selected checkpoints when provisioned; see CURRENT_VERSION.md for server path gaps. |
| `results/` | Formal results and their original validation/source evidence; availability varies by host. |

The internal runner retains explicit external methods in addition to the
current paper method. The public source release has a smaller registry.
Historical experimental model branches are not restored by this source sync.
Frozen run roots, runtime environments, third-party trees, native binaries,
results and checkpoints are outside the source synchronization whitelist.
In particular, S1's frozen A `otherpolicy` directory aliases canonical
`otherpolicy`; matching DCC model/config files must remain no-ops, not be
recursively replaced as duplicate files.
