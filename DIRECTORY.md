# Public source layout

Only the current method path, focused tests, and portable evaluation utilities
are included. Model checkpoints, raw results, native build products, and
third-party repositories stay outside Git.

| Path | Purpose |
| --- | --- |
| `agents/ao_replan.py` | AORePlan adapter and diagnostics |
| `planning/ao_replan_algo.py` | Reverse check, accumulated static A*, and cache handling |
| `planning/aoreplan_branch.py` | Proposal/commit interface shared by SRSLM |
| `learning/epom_encoder.py` | Base-policy encoder required by ARPE |
| `learning/epom_trace_context_actor_critic.py` | Frozen EPOM-L weight loading and shared ARPE support |
| `agents/epom_trace_context.py` | ARPE inference, shared trace and episode reset |
| `learning/epom_trace_multiplier_actor_critic.py` | Selected learned trace actor and critic |
| `agents/arpe.py` | Portable ARPE candidate loader |
| `agents/switcher_core.py` | Wait-only override and Switcher features |
| `agents/switcher.py` | Switcher checkpoint loader and inference |
| `learning/switcher_actor_critic.py` | Two-action Switcher actor-critic |
| `pomapf_env/switcher_arpe_env.py` | Switcher training environment |
| `agents/srslm.py` | Full SRSLM composition |
| `configs/arpe_final_candidate.json` | Default ARPE/base weight paths |
| `maps/train.yaml` | Complete training map registry |
| `maps/test.yaml` | Complete 36-map test registry, including four resized WC3 maps |
| `docs/MAPS.md` | Split definition and MovingAI WC3 provenance |
| `run_experiments.py` | Evaluator and public method registry |
| `train.py` | ARPE training entry point |
| `train_switcher.py` | Current wait-rule Switcher training entry point |
| `learning/train_arpe_final.yaml` | Current native ARPE training example; final inference is configured separately |
| `learning/train_switcher.yaml` | Switcher training example; original curriculum is retained in the run archive |
| `scripts/train_arpe.sh` | ARPE launcher |
| `scripts/train_switcher.sh` | Final Switcher launcher |
| `tests/` | Regression and portable-loading tests |

Historical model branches and private external comparison adapters are not
restored by source synchronization.
