# ARPE naming and checkpoint compatibility

The current paper method is **ARPE (Action Reweight with Policy Entropy)**.
This is the new name of the selected entropy-gated method previously called
CAAR, not a new network or a newly trained checkpoint.

Use `--algorithms ARPE`, `--arpe-candidate-manifest`, and
`--arpe-weights-path` with the current evaluator. The implementation is
`agents/arpe.py`, exposing `ARPE`, `ARPEConfig`, and
`ArpeCandidateArtifact`. The selected declarations are
`configs/arpe_final_candidate.json` and
`configs/arpe_noentropy_candidate.json`. Their contents, checkpoint paths,
and four artifact SHA256 digests are unchanged.

The separate no-entropy ablation is a separately trained model. Renaming the
gated method does not make inference-time gate removal an equivalent training
ablation.

## Names that deliberately remain in saved formats

Existing checkpoints, their original JSON configs, completed results, and
frozen source archives are immutable. They retain CAAR in historical filenames
and method labels. New source must not be attributed to those old runs.
In particular, the released long-horizon data still records the original
`CAAR` label; its original-source verifier reads that evidence unchanged.

The selected Switcher also retains the serialized `caar_action` input,
`switcher_caar_*` settings, candidate kind/schema strings, and routing-counter
keys used in the saved artifact contracts. These are format identifiers, not
the current method name. Changing them would break checkpoint or historical
audit compatibility. They do not change branch order, logits, action sampling,
rewards, or the wait rule.

The old NoReweight backbone is not ARPE. Its common loading code is now
`agents/policy_backbone.py`; its encoder is
`learning/no_reweight_encoder.py`. The saved NoReweight registration
`encoder_custom: caar` and the `caar_num_filters` /
`caar_num_res_blocks` config fields remain readable for the original weights.
The retired tau actor is still rejected.

There is no old CAAR runtime/CLI alias in the current source. Use a frozen
archive to rerun an old source version, or the current ARPE entrypoints with
the same hash-pinned weights to reproduce the selected policy under its new name.
