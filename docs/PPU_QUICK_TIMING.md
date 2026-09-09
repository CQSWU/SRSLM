# Quick joint-action timing with PPU inference

This completed 2026-09-09 check measures the cost of producing one joint action
for the whole team. It contains 48 short episodes and **3,072 measured calls**.
It is not an exact960 experiment, a throughput result, or a full-map-suite
latency benchmark.

## Protocol and hardware

- One map: `random-s40_d0.15`; populations 100, 300 and 600; seed 0.
- Eight methods under `block_both` and `soft`: 8 x 2 x 3 = 48 episodes.
- Every episode creates a fresh model/planner. Its first 16 real decisions
  are warm-up, followed immediately by 64 measured decisions. The environment
  advances normally through all 80 steps, with `on_target=restart` and
  observation radius 5. There is no reset at the measurement boundary.
- All 64 measured times are retained, including slow samples. Across the run,
  there are 768 warm-up calls and 3,072 measured calls; warm-up is excluded
  from the published timing distributions.
- A single sequential worker is bound to logical CPU 1. Torch intra-op and
  inter-op threads, and the configured BLAS/OpenMP thread limits, are one.
- One visible **PPU-ZW810E**, physical device 0, is used through the backend's
  `cuda:0` API. Python is 3.10.13 and Torch reports version 2.4.0.
- EPOM-L, ARPE, CHS, PRIMAL2, SRSLM and DCC-L were checked for actual neural
  parameter placement on `cuda:0`; SRSLM's ARPE and Switcher components were
  both checked. Their preprocessing and search work can still execute on CPU.
  AORePlan and PIBT are pure CPU planners in this check.
- The selected paper checkpoints are unchanged between collision rules.
  ARPE/SRSLM retain the selected block-trained models; DCC-L uses the selected
  update-100 reward-safe checkpoint, not the failed longer fine-tune.

The timer surrounds `algo.act()` for the entire team. For learned methods,
device synchronization finishes prior work **before** the clock starts, and
a second synchronization completes this call's work **before** the clock
stops. The latter wait is included. CPU planners do not invoke device
synchronization. Environment stepping, model loading, model reset and the
pre-call synchronization are outside the interval. Any preprocessing, search
or transfer performed inside `act()` remains inside it.

These are wall-clock milliseconds per **joint action**, not milliseconds per
agent. No division by population is applied.

## Mean results

Each cell is the mean of all 64 measured calls for that method/rule/population
tuple. Display values are rounded to two decimals; CSVs retain the original
full-precision values. Lower is faster.

| Method | Both: 100 | Both: 300 | Both: 600 | Soft: 100 | Soft: 300 | Soft: 600 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EPOM-L | 5.85 | 13.13 | 24.71 | 5.80 | 13.27 | 25.17 |
| AORePlan | 19.09 | 92.36 | 206.67 | 20.52 | 91.15 | 212.83 |
| ARPE | 16.59 | 38.46 | 71.19 | 16.68 | 38.71 | 71.85 |
| CHS | 28.34 | 44.42 | 55.04 | 30.18 | 50.46 | 64.51 |
| PRIMAL2 | 80.22 | 236.34 | 474.14 | 80.24 | 240.28 | 483.49 |
| SRSLM | 43.66 | 159.50 | 325.82 | 46.34 | 158.12 | 330.41 |
| PIBT | 12.78 | 26.26 | 32.82 | 14.44 | 34.22 | 58.41 |
| DCC-L | 41.59 | 81.50 | 136.22 | 57.26 | 123.01 | 173.95 |

## Complete timing data

- [All 48 means, medians and P95 values](data/ppu_joint_action_timing_20260909.csv)
- [All 3,072 individual measured times](data/ppu_joint_action_samples_20260909.csv)

The sample table preserves episode steps 17 through 80 and the integer
nanosecond durations, as well as their millisecond conversions. For each
group, the median is the midpoint of the two central sorted values. P95 uses
linear interpolation at index `0.95 * (64 - 1)` in zero-based sorted order.
No trimming, fastest-run selection or outlier removal is applied.

| CSV | SHA-256 |
| --- | --- |
| Timing summaries | `85ca5a02cddd8dc4d0ef332d24fc84bc58f99e211f94f6aae14075659ad80f02` |
| Individual samples | `62c87f1f03caa50f91226a59ba453d0e97126280e34d3dcc8e50b520c853b528` |

The public CSVs are exact copies of the accepted collection. The retained
run passed raw/journal agreement, unique-grid, 80-call partition, all-sample
summary, checkpoint, device and declared-source stability checks. Private
execution paths, nested runtime records and weights are not published here.

## Interpretation limits

The physical host is shared, so this is not an isolated-hardware benchmark.
One map, one seed and 64 measured steps per tuple do not establish steady-state
latency, scaling complexity or lifelong throughput. This is a whole-`act()`
measurement, not a profile separating neural inference, trace processing,
search and transfer costs.

The [earlier CPU-only quick check](QUICK_TIMING.md) is preserved separately.
It included the first call among 16 measured steps, whereas this check uses
16 warm-up calls and then 64 measured steps, in addition to changing neural
inference device. The resulting ratios must not be presented as pure hardware
acceleration factors or as a controlled CPU-versus-PPU ablation.
