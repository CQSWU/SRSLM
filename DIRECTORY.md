# Public source layout

Only the current method path, focused tests, and portable evaluation utilities
are included. Model checkpoints, raw results, native build products, and
third-party repositories stay outside Git.

| Path | Purpose |
| --- | --- |
| `agents/replan.py` | RePlan adapter |
| `planning/replan_algo.py` | Dynamic replanning and original no-path fallback |
| `agents/ao_replan.py` | AORePlan adapter and diagnostics |
| `planning/ao_replan_algo.py` | Reverse check, accumulated static A*, and cache handling |
| `planning/aoreplan_branch.py` | Proposal/commit interface shared by SRSLM |
| `agents/epom.py` | EPOM/EPOM-L inference backbone |
| `agents/epom_direct_reweight.py` | Parameter-free Direct trace correction |
| `agents/epom_trace.py` | Shared trace state and episode reset |
| `agents/epom_trace_context.py` | Selected ARPE inference adapter |
| `learning/epom_trace_multiplier_actor_critic.py` | Selected learned trace actor and critic |
| `agents/arpe.py` | Hash-pinned ARPE candidate loader |
| `agents/switcher_core.py` | Shared wait-aware routing and Switcher features |
| `agents/switcher.py` | Switcher checkpoint loader and inference |
| `learning/switcher_actor_critic.py` | Two-action Switcher actor-critic |
| `pomapf_env/switcher_arpe_env.py` | Switcher training environment |
| `agents/srslm.py` | Full SRSLM composition |
| `agents/srslm_arpe_ablation.py` | NoWait and OnlyWait ablations |
| `configs/arpe_final_candidate.json` | Selected ARPE/base artifact identities |
| `maps/eval_capacity_intersection_n600.yaml` | Original 32-map evaluation registry |
| `maps/eval_wc3_extra3.yaml` | Three added resized WC3 maps |
| `maps/eval_wc3_extra1_timbermawhold.yaml` | Fourth added resized WC3 map |
| `maps/movingai_wc3_512/` | Added map files and source records |
| `run_experiments.py` | Evaluator with a restricted public method registry |
| `train.py` | EPOM-L and ARPE training entry point |
| `train_switcher_wait_arpe.py` | Wait-aware Switcher training entry point |
| `train_switcher_nowait.py` | NoWait Switcher training entry point |
| `tests/` | Regression and artifact-contract tests |

Historical model branches and private external comparison adapters are not
restored by source synchronization.
