# Method names and saved formats

ARPE means **Action Reweight with Policy Entropy**. Use `--algorithms ARPE`
with the evaluator. `configs/arpe_final_candidate.json` supplies the current
relative checkpoint paths and inference settings; another compatible
declaration can be supplied through `--arpe-candidate-manifest`.

The current method includes both a selected checkpoint and its inference
configuration. Changing the entropy threshold, residual multiplier or fixed
Direct bonus changes the evaluated policy even if the checkpoint is unchanged.
See [CURRENT_VERSION.md](../CURRENT_VERSION.md) for the exact retained version.

Original checkpoint files keep their saved parameter names.
The Switcher still uses serialized `caar_action`, `switcher_caar_*` and candidate
schema fields. These are saved-format identifiers, not additional methods.
Removing them would prevent loading the retained weights. Legacy experiment
sources belong to the separate historical archive, not the current default path.

Public entries are AORePlan, ARPE and SRSLM. Standalone baselines and ablation
implementations are archived outside this source tree. Existing experiment
records retain their original method names and configurations.
