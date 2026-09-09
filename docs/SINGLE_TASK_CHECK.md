# Small single-task check

This supplementary check, completed on 2026-09-09, evaluates whether agents
reach their initial goals within a fixed time limit. It is **240 episodes in
total**, not exact960, a lifelong throughput experiment, or a latency benchmark.

## Protocol

- Methods: RePlan, AORePlan, ARPE, SRSLM and PIBT.
- Rules: `block_both` and `soft`.
- Maps: `mazes-s40_wc4_od30`, `random-s40_d0.15`, `sc1-TheFrozenSea` and
  `street-Shanghai_0`.
- Populations: 100, 300 and 600; seeds: 0 and 42; observation radius: 5.
- POGEMA `on_target=finish`, with a 512-step limit. An agent disappears on
  reaching its goal and receives no replacement goal. A fully successful
  episode can end before the limit.
- The same map/population/seed tuple has identical initial positions and
  targets across methods and rules. Every episode creates and resets fresh
  policy/planner state.
- ARPE and SRSLM use the selected **block-trained** models identified in
  [Current verified version](../CURRENT_VERSION.md), unchanged under soft.
  There was no single-task retraining or inference-time entropy override.
- Default AORePlan retains its conservative static-step occupancy check under
  both rules. This is not the separate `AORePlan-SoftNoCheck` search ablation.

The grid is 4 maps x 3 populations x 2 seeds x 5 methods x 2 rules = 240.
Each method/rule has 24 episodes, or eight episodes at each population.
The separate ten-episode and four-worker smoke checks are excluded.

## Results

**ISR** is the fraction of initial agents that reach their goals in an episode.
**CSR** is 1 if every initial agent succeeds, otherwise 0. Both are averaged
with equal episode weight, not pooled by agent count. The table uses percentages;
the downloadable CSV files retain full-precision fractions from 0 to 1.

| Method | Both ISR | Both CSR | Soft ISR | Soft CSR |
| --- | ---: | ---: | ---: | ---: |
| RePlan | 96.61% | 75.00% | 97.99% | 83.33% |
| AORePlan | 94.30% | 8.33% | 98.73% | 54.17% |
| ARPE | 97.44% | 33.33% | 99.62% | 54.17% |
| SRSLM | 99.67% | 58.33% | 99.67% | 62.50% |
| PIBT | 86.09% | 45.83% | 99.92% | 87.50% |

SRSLM has the highest mean ISR under block_both, but RePlan has the highest
CSR under that rule. PIBT has the highest ISR and CSR under soft. Thus neither
SRSLM nor any other method leads both metrics under both rules. These are
descriptive results from a small, fixed map set, not a general ranking.

PIBT is a **centralised, full-information reference**, not a partially
observable peer. Its finish adapter removes completed agents from the active
solver inputs while retaining surviving priorities and RNG state. Under
block_both, the simulator rejects following moves that standard PIBT relies on;
that entry is a constrained-execution check, not a claim that its usual
guarantees hold. It does not fill PIBT's omitted block_both cell in the main
lifelong throughput comparison.

For ARPE/SRSLM, completed agents stop depositing trace. Success is checked from
one-time arrival events, with no goal reassignment. Auxiliary switching counts
include inactive batch entries and are not used as active-agent routing rates.

## Downloadable data and scope

- [All 240 episode records](data/single_task_episodes_20260909.csv)
- [Thirty population-level summaries](data/single_task_by_population_20260909.csv)
- [Ten overall summaries](data/single_task_overall_20260909.csv)

These are byte-for-byte copies of the accepted collection outputs. The episode
table includes method, rule, map, population, seed, protocol, actual episode
length, completed-agent count, ISR and CSR. No unsuccessful episode is removed.

| CSV | SHA-256 |
| --- | --- |
| Episode records | `5ba125f47f5da6426b4f1409d79af65adea56c95dc40696fc038ee5cad336ace` |
| Population summaries | `9ce61c0652972b60efcb0f32e6df299308a3847a5fdbc9a0484a957569d72bec` |
| Overall summaries | `8a2ae3ad815e4af24884ef5d3499c7e3cd7a1d4dcb366bd42f3de6303fb45061` |

The retained run passed complete-grid, paired-placement, raw/journal,
one-time-success and selected-checkpoint checks; its 189 declared source and
artifact hashes remained unchanged. Public CSVs support independent metric
aggregation. Exact execution additionally requires the retained finish-evaluation
adapters and separately supplied checkpoints. This result update does not
publish private execution files, third-party implementations or weights, and
does not claim that this public checkout alone reproduces the entire run.
