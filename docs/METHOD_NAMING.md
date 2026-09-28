# Method names and saved formats

ARPE means **Action Reweight with Policy Entropy**. Use `--algorithms ARPE`
with the evaluator. `configs/arpe_final_candidate.json` supplies the current
relative checkpoint paths and inference settings; another compatible
declaration can be supplied through `--arpe-candidate-manifest`.

The current method includes both a selected checkpoint and its inference
configuration. Changing the entropy threshold, residual multiplier or fixed
Direct bonus changes the evaluated policy even if the checkpoint is unchanged.
See [CURRENT_VERSION.md](../CURRENT_VERSION.md) for the exact retained version.

Original checkpoint/config/result files keep their original names and hashes.
The Switcher still uses serialized `caar_action`, `switcher_caar_*` and candidate
schema fields. These are saved-format identifiers, not additional methods.
Removing them would prevent loading the retained weights. Legacy experiment
sources belong to the separate historical archive, not the current default path.

Current switching comparisons are Full, NoRule and OnlyRule. NoRule uses the
same full Switcher with both overrides disabled. OnlyRule applies the wait and
reverse overrides without a Switcher. Older independently trained NoWait
results must not be relabeled as this matched NoRule experiment.
