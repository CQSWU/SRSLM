# ARPE reweighting ablation

The completed comparison has five configurations under each of `block_both`
and `soft`. Each configuration/rule covers 32 held-out maps, populations
100/200/300/400/500/600 and seeds 0/42/123/2024/3407: 960 episodes, each with
512 steps, observation radius 5 and lifelong `restart`.

## Configurations

- **Base policy (EPOM-L):** the selected frozen lifelong base policy.
- **Direct reweight:** fixed signed trace-pressure correction at every step.
- **Direct + policy entropy:** the same fixed correction, enabled by the
  policy-entropy gate. Both Direct arms center over the free cells of the
  entire 11x11 crop, not just five candidate cells. Neither uses clipped ReLU.
- **ARPE without policy entropy:** a separately trained 500-million-step
  ungated trace branch, not the gated model with its gate disabled at test time.
- **ARPE:** the selected learned trace branch trained and evaluated with
  policy entropy. The ungated training retains the same architecture, frozen
  EPOM-L base, training maps/population, frame budget and PPO settings.

Each configuration uses the same selected block-trained weights under both
execution rules, with no soft-specific training. See the
[artifact identities](../CURRENT_VERSION.md) and
[evaluation settings](../README.md#arpe-and-direct-ablations).

## Overall throughput

Throughput is completed goals per environment step. Each cell averages all
960 episodes with equal weight. The display uses four decimals; the downloadable
population-level CSV keeps the original full precision.

| Configuration | block_both | soft |
| --- | ---: | ---: |
| Base policy (EPOM-L) | 1.4308 | 2.3368 |
| Direct reweight | 1.4323 | 2.7081 |
| Direct + policy entropy | 1.4857 | 2.5641 |
| ARPE without policy entropy | 1.4998 | 2.1021 |
| ARPE | 1.5467 | 2.4353 |

ARPE's 1.5467 is the highest overall mean under block_both. Under soft,
Direct reaches 2.7081, above Direct + entropy at 2.5641 and ARPE at 2.4353.
The learned branch behaves differently, with gated ARPE outperforming the
independently trained ungated branch under both rules. Neither entropy
gating nor learning the correction improves every tested comparison.

## Data and evidence boundaries

[Download all 60 population-level points](data/arpe_both_soft_ablation_20260908.csv).
Each point averages 160 episodes (32 maps x five seeds). The CSV is an
unchanged copy of the accepted output, SHA-256
`543780f5d7e2211b19115bf222a5824fb5256da32a75bdf8016a94cfd2568301`.
It retains historical labels: `CAAR + policy entropy` is displayed as ARPE,
and `CAAR without policy entropy` as ARPE without policy entropy. Renaming
does not change the model, rows or values.

The retained numerical audit is `CAAR_BOTH_SOFT_AUDIT.json`, SHA-256
`08d8ce3c801db7eb7270fb4e01318bd1ce5bbcf942696cdefa8575ed416d338b`.
It binds all ten 960-episode arms and the 60 published points, but their
original evidence is not uniform:

- The four newly completed non-baseline soft arms and the controlled
  block_both Direct + entropy arm have original result/journal, execution
  contract, source/dependency checks and transfer evidence.
- The recovered soft EPOM-L baseline has historical raw/source hashes and
  a post-hoc audit, but no original validation file. Missing original
  journal or runtime-source evidence is not reconstructed or claimed.
- Retained block_both arms are numerically tied to their unchanged raw files
  and original package audits. Where historical journal or runtime-source
  evidence was not packaged, this release does not claim to supply it.

The current block_both Direct + entropy value, 1.4856831868489584, comes from
the new controlled full-crop, signed experiment. It replaces the historical
1.4837666829427083 value in this comparison. That older result did not bind
its centering convention or original runtime source, so it cannot certify
a gate-only comparison with current Direct. The old records and limitations
remain retained; they are not silently repaired or assigned to the new control.
The plain block_both Direct rows match the separately recorded explicit
full-crop, ungated, signed run.

This public update supplies aggregate data and the evidence identifiers,
not private raw bundles, machine paths or checkpoints. It does not claim
that every historical arm has a complete original runtime-source snapshot.
