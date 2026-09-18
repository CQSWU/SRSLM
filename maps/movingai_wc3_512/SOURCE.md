# MovingAI WC3-512 source

The raw maps in this directory come from the official MovingAI Warcraft III
512x512 benchmark archive:

- Index: https://www.movingai.com/benchmarks/wc3maps512/index.html
- Archive: https://www.movingai.com/benchmarks/wc3maps512/wc3maps512-map.zip
- Archive SHA256: `12ef51dea736c0dedfd4fc0a04234751af27c7247cb4ae53e9d0ff467976c17a`

The fixed evaluation subset contains the three maps with the largest reported
numbers of traversable states: Dustwallow Keys, Icecrown, and Plunder Isle.
Each complete raw map is reduced to 128x64 without cropping. One output cell
represents a 4x8 source block and is walkable when at least half of that block
is walkable. This area-voting rule preserves the full-map layout while avoiding
interpolated terrain symbols. `scripts/build_movingai_wc3_test3.py` verifies the
raw hashes and state counts and performs the deterministic conversion.
