# Map split

The public project uses exactly two map registries:

- `maps/train.yaml` contains the 186 training maps.
- `maps/test.yaml` contains the 36 evaluation maps: the fixed 32-map
  capacity-intersection set and four resized WC3 maps.

Every registry entry stores its grid directly. Evaluation therefore needs no
secondary name list, source directory, or fallback registry.

The four added maps are `DustwallowKeys`, `Icecrown`, `PlunderIsle`, and
`TimbermawHold` from the official MovingAI Warcraft III 512x512 benchmark.
Each full map was reduced to 128x64 without cropping: one output cell represents
a 4x8 source block and is walkable when at least half of that block is walkable.

- Index: https://www.movingai.com/benchmarks/wc3maps512/index.html
- Archive: https://www.movingai.com/benchmarks/wc3maps512/wc3maps512-map.zip
- Archive SHA256: `12ef51dea736c0dedfd4fc0a04234751af27c7247cb4ae53e9d0ff467976c17a`

The split is fixed. Training and evaluation commands should reference these
two files directly rather than creating experiment-specific map lists.
