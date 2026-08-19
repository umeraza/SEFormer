# Reproducibility audit

This audit separates statements in the supplied manuscript from implementation
choices required to make executable code. It is intentionally candid: a public
repository should not imply exact reproduction when essential experimental
metadata are absent.

## Directly specified by the manuscript

| Item | Paper value | Repository mapping |
| --- | --- | --- |
| Input face size | `112 x 112` | `data.image_size: [112, 112]` |
| Default sampled frames | 32 | `data.frames: 32` |
| Default temporal stride | 1 (selected ablation) | `data.temporal_stride: 1` |
| Views | fine, mid, coarse | three `model.views` entries |
| 3D tubelets | `(2,8,8)`, `(4,8,8)`, `(8,8,8)` | `model.views[*].patch_size` |
| Inter-view operation | coarser query, adjacent finer key/value | `fusion.direction: coarse_to_fine` |
| View aggregation | learned sequence pooling | `model.pooling: sequence` |
| Global encoder | Transformer, selected depth 4 and width 256 | `model.global.*` |
| Classifier | MLP | `model.classifier_hidden` |
| Optimizer | AdamW | `training.optimizer: adamw` |
| Schedule | cosine decay | `training.scheduler: cosine` |
| Loss | cross-entropy | `training.loss: cross_entropy` |
| Regularizers named | label smoothing, stochastic depth, Gaussian noise, vertical flip | explicit config fields |
| DAiSEE task | four engagement-intensity levels | four labels, ordered 0--3 |
| BAUM-1s task | six basic emotions | six normalized labels |
| Reported metrics | accuracy, class-averaged precision/recall/F1 | accuracy, macro P/R/F1, and table-compatible `paper_f1` |

## Necessary inferred defaults

The following values are **not present in the SEFormer manuscript**. They are
chosen so the code is runnable and are never presented as recovered facts.

| Item | Default | Rationale / risk |
| --- | --- | --- |
| View width / MLP / depth | 512 / 1024 / 3 | Matches the closest EngageFormer configuration; may differ from authors' run |
| View attention heads | 8 | Standard divisor of 512; EngageFormer reports 3 heads with width 512, which is incompatible with equal-width standard heads |
| Global MLP width | 1024 | Makes the paper-aligned model close to the reported 28.5M parameter scale |
| Global heads | 8 | Valid divisor of width 256 |
| Initial learning rate | `1e-4` | Reported by EngageFormer, not by the supplied SEFormer text |
| Weight decay | `1e-5` | Reported by EngageFormer, not by the supplied SEFormer text |
| Epochs | 100 | Reported by EngageFormer, not by the supplied SEFormer text |
| Batch size / accumulation | 2 / 4 | Conservative memory-oriented default; performance-sensitive |
| Label smoothing | 0.1 | Conventional nonzero value; performance-sensitive |
| Dropout / stochastic depth | 0.1 / 0.1 | Conventional regularization values |
| Gaussian-noise standard deviation | 0.01 after scaling to `[0,1]` | Magnitude is unreported |
| Vertical-flip probability | 0.5 | Probability is unreported; vertical face flips are scientifically questionable and should be ablated |
| Normalization | mean/std `0.5` per channel | Pretraining and normalization are unreported |
| Checkpoint criterion | validation macro-F1 | Suitable for class imbalance but unreported |
| Random seeds | 42 by default, configurable repeats | No seeds or repeated-run protocol are reported |
| CVAF placement | after all per-view encoder blocks | Follows the manuscript algorithm; prose/figure also suggest intermediate placement |
| Sequence pool domain | class token plus patch tokens | “all tokens” is ambiguous; configurable via `pool_include_cls` |

Every inferred field can be changed with a YAML edit or `--set dotted.key=value`.

## Internal inconsistencies requiring author resolution

1. **Class count.** Methodology states `K=6` for the general problem, but DAiSEE
   is a four-class task. The repository binds class count to each dataset.
2. **Multiclass metric formulas and F1 aggregation.** The paper writes binary
   TP/TN equations while evaluating four- and six-class problems. Moreover, its
   reported F1 values equal the harmonic mean of reported macro precision and
   macro recall. That is not generally equal to the conventional macro-F1 (mean
   of class-wise F1). Code reports both as `paper_f1` and `macro_f1`.
3. **Token count.** For 32 frames and 112x112 images, non-overlapping
   `(2,8,8)`, `(4,8,8)`, `(8,8,8)` patches produce `3136 + 1568 + 784 = 5488`
   patch tokens (plus three CLS tokens). The ablation table reports 1288. The
   same equation yields 8,624, 21,952, and 1,372 tokens for the temporal-dense,
   spatial-fine, and spatial-coarse schedules, whereas the table reports 1,782,
   2,564, and 332. None of the described operations reconciles these totals.
4. **BAUM-1 size and labels.** The manuscript reports 1,222 BAUM-1s clips and
   describes eight emotions plus four mental states. The official UCI record
   reports 1,184 clips and 13 states, including `neutral` and `confusion`.
   Repository preparation is annotation-driven and prints observed counts.
5. **BAUM-1 split.** No train/validation/test or cross-validation protocol is
   stated. Results cannot be compared exactly until the authors publish clip or
   subject assignments.
6. **CVAF direction.** The abstract says “bidirectional information exchange,”
   while the equation and algorithm update only the coarser view from the finer
   view. The default follows the equation; `bidirectional` is available.
7. **Table 5 caption.** It calls the 2D-vs-3D experiment a BAUM-1 ablation, but
   its numbers exactly match the DAiSEE full-model row. The config names it as a
   DAiSEE ablation pending clarification.
8. **Global width comparison.** The depth/width table changes depth and width
   together for the final row, so it cannot isolate the effect of width.
9. **Model novelty.** The manuscript itself acknowledges substantial overlap
   with EngageFormer. This code preserves the stated SEFormer formulation but
   does not make a software-level claim of architectural priority.

## Protocol safeguards in this repository

- Official DAiSEE split names are preserved; the validator checks subject
  overlap and duplicate clips.
- BAUM fallback splits are subject-disjoint and stamped as generated.
- The resolved configuration, Python/package environment, seed, Git commit (if
  available), per-sample predictions, and class ordering are saved with a run.
- Metrics always receive the complete configured label set, including classes
  with zero predicted samples (`zero_division=0`).
- Test data are evaluated only by explicit command; training selects on the
  validation split.
- Confusion matrices are saved as raw counts and row-normalized values.
- Dataset paths and licensed media are excluded from Git.

## Minimum information still needed from the authors

For a defensible exact-reproduction release, publish:

1. exact DAiSEE and BAUM clip/subject split files;
2. view widths, MLP ratios, heads, and layer depths;
3. batch size, learning rate, weight decay, warmup, epochs, gradient clipping,
   and checkpoint-selection rule;
4. augmentation magnitudes and probabilities, normalization, and face-crop
   failure policy;
5. initialization/pretraining details and all random seeds;
6. metric averaging mode and whether validation tuning touched the test set;
7. the operation that reduces the claimed token total to 1,288;
8. repeated-run mean/standard deviation or confidence intervals.

Until these are resolved, this repository is a faithful, testable
**implementation specification**, not evidence that the headline scores have
been independently reproduced.
