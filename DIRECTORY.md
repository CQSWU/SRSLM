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
| `agents/arpe.py` | Portable ARPE candidate loader |
| `agents/switcher_core.py` | Shared wait-aware routing and Switcher features |
| `agents/switcher.py` | Switcher checkpoint loader and inference |
| `learning/switcher_actor_critic.py` | Two-action Switcher actor-critic |
| `pomapf_env/switcher_arpe_env.py` | Switcher training environment |
| `agents/srslm.py` | Full SRSLM composition |
| `agents/srslm_arpe_ablation.py` | NoWait and OnlyWait ablations |
| `configs/arpe_final_candidate.json` | Default ARPE/base weight paths |
| `maps/train.yaml` | Complete training map registry |
| `maps/test.yaml` | Complete 36-map test registry, including four resized WC3 maps |
| `docs/MAPS.md` | Split definition and MovingAI WC3 provenance |
| `run_experiments.py` | Evaluator and public method registry |
| `train.py` | EPOM-L and ARPE training entry point |
| `train_switcher.py` | Unified final/NoWait Switcher training entry point |
| `learning/train_epom.yaml` | EPOM-L training recipe |
| `learning/train_arpe.yaml` | Selected ARPE training recipe |
| `learning/train_switcher.yaml` | Final Switcher training recipe |
| `scripts/train_epom.sh` | EPOM-L launcher |
| `scripts/train_arpe.sh` | ARPE launcher |
| `scripts/train_switcher.sh` | Final Switcher launcher |
| `tests/` | Regression and portable-loading tests |

Historical model branches and private external comparison adapters are not
restored by source synchronization.
