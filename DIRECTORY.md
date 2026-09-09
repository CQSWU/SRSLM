# Public source layout

Only the current method path, focused tests, and audited launch/validation
utilities are included. Model checkpoints, result files, logs, generated
binaries, and third-party repositories stay outside Git.

| Path | Purpose |
| --- | --- |
| `agents/replan.py` | RePlan adapter |
| `planning/replan_algo.py` | RePlan dynamic planning and original no-path fallback |
| `agents/ao_replan.py` | AORePlan adapter and diagnostics |
| `planning/ao_replan_algo.py` | Reverse detection and static-map A* check |
| `planning/aoreplan_branch.py` | Proposal/commit interface shared by SRSLM |
| `agents/direct.py` | Historical independent-backbone fixed reweighting |
| `agents/epom_direct_reweight.py` | Paper EPOM-L Direct, with explicit entropy/transform/centring options |
| `agents/ao_replan_soft_ablation.py` | Standalone soft search exception; not used inside SRSLM |
| `agents/epom_trace_context.py` | Current ARPE inference adapter |
| `learning/epom_trace_multiplier_actor_critic.py` | Current learned trace branch and independent critic |
| `agents/arpe.py` | Hash-pinned frozen ARPE candidate loader |
| `agents/switcher_core.py` | Shared wait-aware routing and Switcher state construction |
| `agents/switcher.py` | Switcher checkpoint loader and inference |
| `learning/switcher_actor_critic.py` | Feed-forward two-branch actor-critic |
| `learning/switcher_learner_patch.py` | PPO masking for states where Switcher acts |
| `pomapf_env/switcher_arpe_env.py` | Wait-aware Switcher training environment |
| `agents/srslm.py` | Deployed SRSLM composition |
| `agents/srslm_arpe_ablation.py` | NoWait and OnlyWait with the paper's pinned ARPE |
| `agents/switcher_arpe.py` | Independently trained NoWait Switcher loader |
| `run_experiments.py` | Evaluation runner with a restricted public method allowlist |
| `train.py` | Shared Sample Factory training entry point |
| `train_switcher_wait_arpe.py` | Wait-aware Switcher training entry point |
| `train_switcher_nowait.py` | Independently trained all-state Switcher entry point |
| `configs/arpe_final_candidate.json` | Public path-relative ARPE/base checkpoint identities |
| `learning/*.yaml` | Current EPOM-L, ARPE, and Switcher smoke/formal recipes |
| `maps/eval_capacity_intersection_n600.yaml` | Exact960 32-map list |
| `scripts/run_srslm_wait_aware_arpe_100m_exact960_server1.sh` | Hash-pinned formal SRSLM launcher |
| `scripts/validate_srslm_wait_aware_arpe_100m_exact960.py` | Formal result and artifact validator |
| `tests/` | Focused regression and artifact-contract tests |

The old dual absolute-return-estimator selector and its data-collection,
training, and compatibility files were removed. External comparison
repositories and server-only experimental branches are not vendored here.
