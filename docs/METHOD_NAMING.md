# ARPE naming and checkpoint compatibility

The current paper method is **ARPE (Action Reweight with Policy Entropy)**.
This is the new name of the selected entropy-gated method previously called
CAAR, not a new network or a newly trained checkpoint.

Use `--algorithms ARPE` with the current evaluator. Set weight paths through
`--arpe-candidate-manifest` when needed. The implementation is
`agents/arpe.py`, exposing `ARPE`, `ARPEConfig`, and
`ArpeCandidateArtifact`. The selected declarations are
`configs/arpe_final_candidate.json`. The declaration contains portable paths;
hash values are recorded after loading only as optional provenance.

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

The old NoReweight method and its encoder are no longer executable. A narrow
checkpoint-config sanitizer still drops inert historical fields while loading
the selected frozen artifacts; it cannot register or construct the retired
model. The retired tau actor is also rejected.

There is no old CAAR runtime/CLI alias in the current source. Use a frozen
archive to rerun an old source version, or the current ARPE entrypoints with
the same released weights to reproduce the selected policy under its new name.
