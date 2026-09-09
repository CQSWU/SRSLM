# Switcher ablation results

The completed comparison uses 32 held-out maps, populations
100/200/300/400/500/600, seeds 0/42/123/2024/3407, lifelong restart,
512 steps and observation radius 5. Each entry averages 960 episodes.
Throughput is completed targets per simulator step. The same selected
block-trained weights are evaluated under both rules.

| Configuration | block_both throughput | soft throughput |
| --- | ---: | ---: |
| OnlyWait | 1.581006 | 2.317405 |
| NoWait | 1.842916 | 2.562828 |
| Full SRSLM | 1.860878 | 2.644157 |

OnlyWait uses ARPE when AORePlan returns wait and AORePlan otherwise, without
a learned Switcher. NoWait invokes its Switcher at every timestep. Full
SRSLM combines the wait rule with learned routing. All three share the same
gated ARPE. The conservative static-step occupancy check is retained under
both execution rules.

Full SRSLM improves mean throughput over OnlyWait by 17.70% under
block_both and 14.10% under soft. Against NoWait, the gains are 0.97% and
3.17%. These compare separately trained configurations, not the isolated
effect of toggling the wait rule on a fixed Switcher checkpoint.

There is a small low-population exception: under soft at 100 agents,
OnlyWait averages 1.443994, Full SRSLM 1.435437, and NoWait 1.426514.
Full SRSLM leads the three configurations at the other soft populations
and at all six block_both populations.

## Evidence

The six-arm independent audit verifies 960 unique, finite, error-free
map-population-seed tuples in each arm, protocol identity, selected artifacts
and the original result sources. Newly completed soft OnlyWait and NoWait
runs also have original per-episode journals, stable before/after source
and environment records, and cross-host byte verification. The retained
historical Full-soft run has no original journal; its raw rows and source
provenance were checked, and that limitation remains recorded.

The original audit and CSV are retained with the private experiment bundle:

- `INVENTORY_AUDIT.json` SHA256:
  `c0b34060885c1ff63d7b9e125e077b2300b20ff2bf8311fb87248ce1062ce669`
- `SWITCHER_TWO_PROTOCOL_FINAL.csv` SHA256:
  `b0e53af89cf9fa43d6f4a343fed56529d74e5582cd3d2f1ce9c1f7dd7384c957`

These identifiers bind the reported summary to those files; the files and
checkpoints are not included in this public source release. See
[reproducibility requirements](REPRODUCIBILITY.md) before running a new
evaluation. No decision-latency or long-horizon result is claimed here.
