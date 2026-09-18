# MovingAI WC3 fourth-map extension

- Official index: https://www.movingai.com/benchmarks/wc3maps512/index.html
- Official archive: https://www.movingai.com/benchmarks/wc3maps512/wc3maps512-map.zip
- Archive SHA256: `12ef51dea736c0dedfd4fc0a04234751af27c7247cb4ae53e9d0ff467976c17a`
- Selected source: `timbermawhold.map`
- Source SHA256: `280c25f71638e1ca330a430ba7f76ed4a54152956deab5b58bc5c61f07c2cd3e`
- Source dimensions and free cells: 512×512, 149,744
- Registry name: `wc3-128x64-TimbermawHold`
- Resized dimensions and free cells: 128×64, 5,101

Timbermaw Hold has the fourth-largest reported number of traversable states in
the official WC3-512 archive, immediately after the three maps already used.
The complete source map is reduced without cropping. Each output cell represents
a 4×8 source block and is walkable when at least half of that block is walkable.
The resulting free space is one four-neighbor connected component.

