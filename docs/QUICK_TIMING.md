# Quick joint-action timing

This is the short timing comparison requested on 2026-09-09, not a full
map-suite speed benchmark. It measures one complete `algo.act()` call for
the whole team, rather than episode time or time divided by the number of
agents. Environment stepping, model construction and reset are excluded.

## Protocol

- One map: `random-s40_d0.15`; populations 100, 300 and 600; seed 0.
- Eight methods under both `block_both` and `soft`: 48 short episodes.
- A fresh model for every episode; 16 measured decisions including the first
  call, with no warm-up calls discarded: 768 decisions in total.
- One CPU worker, one Torch intra-op and inter-op thread, accelerators
  disabled. The recorded processor is Hygon C86 7490.
- The containers share a physical host. Process and accelerator observations
  were collected, but this is **not an isolated-hardware benchmark**.
- The learned methods use their selected paper checkpoints. DCC-L is the
  early-stopped update-100 checkpoint, not the failed longer fine-tune.

## Results

Mean milliseconds per joint decision; lower is faster. Each cell averages
the 16 calls for its map/population/rule tuple.

| Method | Both: 100 | Both: 300 | Both: 600 | Soft: 100 | Soft: 300 | Soft: 600 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EPOM-L | 219.67 | 674.43 | 1380.81 | 218.50 | 673.11 | 1367.92 |
| AORePlan | 26.72 | 95.57 | 225.39 | 26.02 | 94.36 | 238.14 |
| ARPE | 245.00 | 748.28 | 1517.72 | 241.88 | 738.81 | 1517.06 |
| CHS | 301.05 | 753.15 | 1393.96 | 302.72 | 739.38 | 1409.58 |
| PRIMAL2 | 377.78 | 1130.38 | 2255.37 | 380.76 | 1136.08 | 2262.88 |
| SRSLM | 275.40 | 864.33 | 1749.53 | 276.20 | 868.39 | 1774.14 |
| PIBT | 42.95 | 122.96 | 245.88 | 44.31 | 125.27 | 249.32 |
| DCC-L | 244.84 | 1013.17 | 2866.09 | 240.76 | 982.17 | 2867.22 |

The [full-precision aggregate CSV](data/quick_joint_action_timing_20260909.csv)
retains the raw method identifier `CAAR` alongside its current display name
`ARPE`. This is the same selected model, not an additional method.

These measurements describe short CPU decision costs on this map. They do
not establish a general speed ranking, accelerator performance, steady-state
latency or throughput. PIBT's block-both entry is a timing observation only;
it does not fill the omitted PIBT block-both cell in the main throughput table.

## Retained evidence

The separately distributed private bundle is
`results/extra_A_quick_20260909/raw_and_reproduction_evidence.zip`, SHA-256
`9a55be879d2c429d3f168f86fb19da2e9772ba235320eda7083846924bef01e4`.
Its 112 files include the raw records, journals, source and checkpoint
identities, measurement code and execution observations. The collector
validated all 48 episodes and 768 calls; its `QUICK_VALIDATION.json` SHA-256 is
`4be91030cff3cdc946a72dfe25bbde6f71d69013562a0f4dcade81cbf193ba84`.
Weights remain in their separately retained artifact directories.

The cancelled longer timing run is preserved as historical partial evidence.
Its 128-step rows are not pooled with this completed 16-step comparison.
The public repository supplies the aggregate table, not the private external
adapters, checkpoint files or host-observation bundle.
