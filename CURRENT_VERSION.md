# Current verified version

This file identifies the implementation and artifacts behind the retained
wait-aware SRSLM result completed on 2026-09-03. Checkpoints and result JSON
files are intentionally not committed to Git.

## Method identities

| Name | Role |
| --- | --- |
| `RePlan` | Original dynamic replanning baseline |
| `AORePlan` | RePlan plus a static-map A* check for reverse proposals |
| `EPOM-Lifelong-FT` | Lifelong fine-tuned recurrent base policy, abbreviated EPOM-L |
| `NoReweight` | Base-policy runner without trace correction |
| `Direct` | Fixed capped-ReLU trace correction |
| `CAAR` | Learned entropy-gated five-logit trace correction |
| `Switcher` | Learned categorical selector between CAAR and AORePlan |
| `SRSLM` | Wait-aware composition of CAAR, AORePlan, and Switcher |

## CAAR architecture

The EPOM-L backbone is frozen. The actor-side trace branch receives the whole
aligned 11x11 shared-trace crop after centring over its free cells. Obstacles
and padding remain zero; no action or free-cell mask is given to the learned
branch.

The trace encoder is Conv32 (3x3), two 32-channel residual blocks, and an FC32
projection. Its 32 outputs are concatenated with the frozen 512-dimensional
EPOM-L recurrent state and five base logits. An FC256 layer and a five-output
head produce the learned correction. The correction is applied only when the
base-policy entropy exceeds the configured reference threshold. The separate
critic is a linear value head over the frozen 512-dimensional recurrent state;
its value is not added to the frozen EPOM value. The complete learned branch
has 303,846 trainable parameters.

## Switcher routing

AORePlan and CAAR first produce complete primitive-action candidates. An
AORePlan wait selects CAAR immediately without a Switcher forward pass. A
non-wait AORePlan candidate enters the feed-forward two-branch Switcher, which
samples either CAAR or AORePlan. The controller used for PPO data collection is
the same controller used during evaluation.

## Frozen artifact identities

| Artifact | Frames | SHA-256 |
| --- | ---: | --- |
| EPOM-L checkpoint | 100,016,128 | `f70a305ee68546be95e0a93d7f61c9aec435a50da20624a3b382af2276ad79d2` |
| EPOM-L config | - | `74c5cc0f1c5fdc0043bfcaa2e48e3be9c46c2c652f489a2b83379788e5da69b9` |
| CAAR checkpoint | 500,015,104 | `497118e3aa4fbaecde35e53f31fe3126e11c1a1e5b0b621b89ac0d340002d41b` |
| CAAR config | - | `e76a2b238f196752ec358ce8946eb353caa3a4fe3e4df2a92cf812506d008747` |
| Switcher checkpoint | 100,016,128 | `4973fa420a093e043d2aafb2340863a2be3ad7dda3362ef278a98ef8c1a75185` |
| Switcher policy tensors | - | `c2bd85a0cbcffe49dec8a393e84f022efe9bc8ce916190b497d0571acbb75aa9` |
| Switcher config | - | `de387d7b00f7cb0d56b11d78389d702d301a39fb33a7f3f666189c685e7c0bc6` |
| CAAR candidate manifest | - | `75df038934fd10a71ce5b7e97aca7456546a18940553aa49eb454c89510e654f` |

Expected local paths are:

```text
weights/EPOM-lifelong-finetune-r5/EPOM-Lifelong-Finetune-R5
weights/EPOM-TracePaperConvDirectCorrection-R5-500m/EPOM-TracePaperConvDirectCorrection-R5-S0-20260902
weights/SRSLM-switcher-wait-aware-caar-100m/SRSLM-WaitAware-CAAR-100M
artifacts/caar_final_candidate.json
```

## Validated exact960 result

Protocol: 32 held-out capacity-compatible maps; populations 100, 200, 300,
400, 500, and 600; seeds 0, 42, 123, 2024, and 3407; `block_both`;
lifelong `restart`; 512 steps; observation radius 5; 960 unique episodes.

| Population | Mean throughput |
| ---: | ---: |
| 100 | 1.31678466796875 |
| 200 | 1.91896972656250 |
| 300 | 2.06967773437500 |
| 400 | 2.06068115234375 |
| 500 | 1.96484375000000 |
| 600 | 1.83431396484375 |
| **All** | **1.8608784993489584** |

The validation reports 960 finite error-free rows, a mean congestion rate of
0.3411414636, and an AORePlan-wait bypass rate of 0.1559831659. The result JSON
SHA-256 is
`972a87918e5e2dd5eae2ac4b76c3682c48bb72a00a73df5397d35da828f7c3cc`.

The archived formal code snapshot is
`3cd786dc58a86aa1ad982207d1788fc175e4f93e9c3658b3a7157c3056dd397f`.
Of its 34 non-binary tracked files, 33 are byte-for-byte identical in this
curated tree. The only deliberate difference is `run_experiments.py`, whose
public algorithm allowlist was reduced; the retained algorithm execution and
validation paths were not changed.
