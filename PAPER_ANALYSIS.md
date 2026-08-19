# Critical analysis of the SEFormer manuscript

## Executive assessment

SEFormer is a coherent multi-scale video-classification pipeline, and its main
engineering choices are well matched to the stated problem: short tubelets can
retain brief facial changes, longer tubelets reduce temporal resolution and
emphasize persistent behavior, cross-attention can transfer evidence between
scales, and learned sequence pooling avoids forcing all information through a
single CLS token. The reported component ablations are directionally supportive.

The manuscript is not yet independently reproducible, however, and the current
novelty claim must remain narrow. Its view encoders, adjacent cross-view fusion,
sequence pooling, and global Transformer substantially coincide with
[EngageFormer](https://arxiv.org/abs/2502.10813). The defensible contribution is
therefore the particular SEFormer implementation and its controlled empirical
analysis—not the first use of a three-view Transformer for engagement.

Several numeric and protocol inconsistencies are material: all patch-token
counts conflict with the stated equation and input size; BAUM-1 metadata conflict
with the official record; the BAUM split is absent; metric equations are binary
despite multiclass tasks; the tables compute F1 in a nonstandard way; and many
training/architecture values are omitted. Headline comparisons should be treated
as provisional until exact splits, code, checkpoints, seeds, and repeated-run
uncertainty are released.

## Methodology

### What the model actually specifies

For each face-centered clip, SEFormer creates fine, mid, and coarse token streams
using non-overlapping Conv3D kernels `(2,8,8)`, `(4,8,8)`, and `(8,8,8)`. Each
stream receives a class token and learned position embedding, then passes through
its own pre-normalized Transformer encoder. Views are ordered by token count.
For every adjacent pair, the coarser sequence queries the finer sequence through
CVAF and receives a residual update. Learned sequence pooling converts each view
to one vector. These vectors are projected to a common space, treated as a short
sequence by a global Transformer, pooled again, and classified by an MLP.

This division has a sensible inductive interpretation, but temporal patch size
alone does not guarantee that one branch learns “micro-expression,” “gaze,” or
“posture” semantics. Those labels are hypotheses. Establishing them requires
branch-specific temporal perturbations, attention/feature analyses, or annotated
behavioral events—not only aggregate classification scores.

### CVAF ambiguity

The abstract says CVAF enables “bidirectional information exchange.” The equation
and algorithm update only the coarser stream from the next finer stream. That is
unidirectional coarse-query/fine-context attention. This distinction changes the
graph, parameter count, and representational behavior. The repository follows the
equation by default and supplies an explicit bidirectional control.

CVAF placement is also ambiguous. The prose and encoder figure imply lateral
connections inside or between encoder stages; the algorithm applies fusion only
after all per-view encoders. The implementation follows the algorithm. If fusion
was interleaved in the experiments, layer indices and update ordering must be
published.

### Sequence pooling ambiguity

The manuscript first prepends a CLS token and then says sequence pooling weights
“all tokens.” It does not state whether CLS remains in that pool. The answer
affects both the learned representation and interpretation of the CLS ablation.
The default includes CLS because that is the literal “all tokens” reading; the
choice is configurable.

### Computational implications

With 32 frames at 112x112, the three stated tubelets create:

| View | Grid | Patch tokens |
| --- | --- | ---: |
| Fine `(2,8,8)` | `16 x 14 x 14` | 3,136 |
| Mid `(4,8,8)` | `8 x 14 x 14` | 1,568 |
| Coarse `(8,8,8)` | `4 x 14 x 14` | 784 |
| Total | — | 5,488 |

The fine branch alone has roughly 9.8 million query-key pairs per attention head
per sample. Full attention is therefore activation-memory intensive even though
the parameter count is moderate. Claims that the parallel design is efficient
need measured FLOPs, peak memory, throughput, latency, hardware, batch size, and
comparison at equal accuracy. Token count alone is not an efficiency result.

### Architecture values inferred by the repository

The supplied manuscript gives global depth 4 and width 256 but omits per-view
width, heads, MLP width, and depth. The closest EngageFormer table reports view
width 512, MLP width 1024, and depth 3; its simultaneous report of three heads is
not compatible with standard equal-width multi-head attention because 512 is not
divisible by 3. The repository uses eight heads as an explicit assumption. With
view width 512/depth 3 and global width 256/depth 4, its analytical count is
28.848M—close to the manuscript's 28.5M full-model row, but not proof of the
authors' unpublished configuration.

## Datasets

### DAiSEE

The [official DAiSEE page](https://people.iith.ac.in/vineethnb/resources/daisee/index.html)
supports the manuscript's high-level description: 9,068 clips from 112 users,
four affective dimensions, and four intensity levels. The SEFormer task uses only
the Engagement column, so it is a four-class ordinal-intensity problem—not a
six-class task as the generic methodology statement says.

The discussed test confusion matrix has supports 4, 84, 882, and 814 for Very
Low, Low, High, and Very High. A 50% recall for Very Low is only 2/4 and has huge
uncertainty. Overall accuracy is dominated by High and Very High. Macro measures
help but are themselves unstable for a four-sample class. At minimum, release
the official split manifest, per-sample predictions, repeated-seed variability,
and confidence intervals. Ordinal MAE and quadratic weighted kappa would also
show whether errors remain adjacent.

DAiSEE access requires acceptance of terms, and its page restricts image use and
redistribution. A public code repository must not bundle videos, frames, crops,
or unauthorized example faces. The supplied repository deliberately contains
only manifest-generation code.

### BAUM-1s

BAUM-1s is an affect-recognition benchmark, not a student-engagement benchmark.
Using six basic emotions is a useful complementary stress test, but it does not
establish cross-dataset engagement generalization.

The manuscript reports 1,222 spontaneous clips and describes eight emotional
plus four mental states. The [official UCI record](https://archive.ics.uci.edu/dataset/473/baum%2B1)
reports 1,184 clips from 31 subjects and 13 states, including neutral and
confusion. The exact archive/version and filtering steps must be identified.

More importantly, no BAUM train/validation/test or cross-validation protocol is
given. A random clip split could leak participant identity across partitions and
inflate results. The repository accepts an author-provided split; otherwise it
generates a clearly marked subject-disjoint exploratory split. Such a fallback
cannot reproduce the published number.

The near-reciprocal Disgust/Surprise errors and very low Surprise recall warrant
checking class-index mapping across annotations, classifier output, and plot
labels. That verification should precede architectural interpretation.

## Evaluation metrics

The manuscript gives binary TP/TN formulas for accuracy, precision, recall, and
F1, but evaluates single-label multiclass tasks. Correct reporting requires an
explicit one-vs-rest reduction and averaging rule. The text suggests class-wise
calculation and averaging, which implies macro precision and macro recall.

The published F1 numbers reveal a further issue. For the full DAiSEE row:

\[
2\frac{86.77\times69.83}{86.77+69.83}=77.38.
\]

The same equality holds (within rounding) for the BAUM row and every component
ablation. Thus the table's F1 is the harmonic mean of already averaged precision
and recall. This is not conventional macro-F1, which averages each class's F1.

Reconstructing the DAiSEE confusion counts described in the text gives about
76.29% accuracy, 86.77% macro precision, 69.83% macro recall, **77.38% table F1**,
but only about **76.10% conventional macro-F1**. The repository reports both:

- `paper_f1`: harmonic mean of averaged precision and recall, for table
  comparability;
- `macro_f1`: unweighted mean of per-class F1, for standard multiclass reporting.

Future revisions should name the table metric precisely or replace it with
standard macro-F1 and update comparisons consistently. All metrics must use a
fixed complete label list and define zero-division behavior.

## Experimental setup

The manuscript specifies 32 sampled frames, 112x112 face crops, three tubelets,
AdamW, cosine decay, cross-entropy, label smoothing, stochastic depth, Gaussian
noise, and vertical flipping. It omits:

- per-view width, heads, MLP width, and depth;
- learning rate, weight decay, epochs, batch size, gradient accumulation,
  clipping, warmup, and minimum learning rate;
- initialization or pretraining;
- dropout, stochastic-depth schedule, label-smoothing value, noise magnitude,
  and augmentation probabilities;
- normalization statistics and color convention;
- MTCNN thresholds, margin, tracking, and missing-face behavior;
- exact clip-start sampling, padding, and training-time randomness;
- seeds, number of runs, checkpoint criterion, early stopping, and hardware;
- exact metric averaging and dataset split files.

Vertical flipping turns a face upside down and is not a standard invariance for
webcam engagement. Its inclusion should be justified empirically. The repository
implements it because the paper names it and includes a no-vertical-flip control.

## Ablation studies

### Component removal

The full model leads every reported component row. Removing CVAF reduces
accuracy by 2.55 points but reported F1 by 15.19, largely through a 28.26-point
precision change. Such a disproportionate movement could indicate rare-class
instability, a change in prediction distribution, or metric aggregation effects.
Per-class predictions and multiple seeds are needed before attributing the whole
change to cross-view alignment.

CLS pooling primarily reduces recall, consistent with loss of distributed token
evidence. Removing the global encoder lowers accuracy/F1, and the single-view
model has the largest accuracy loss. These results support usefulness of the
components as a package, but a one-factor removal does not establish synergy or
causality when optimization and parameter counts also change.

### Number of views and parameter matching

The non-matched comparison conflates scale diversity with capacity. The second
comparison is the correct idea, but exact reduced widths/depths and measured
parameter counts are absent. The three-view budgeted row raises recall while
lowering precision, so it supports broader class coverage rather than uniform
superiority. The repository provides analytically budgeted configs and labels
their widths as assumptions.

### Patch schedule

Every reported token total conflicts with the manuscript's own floor-division
equation at the stated input size:

| Schedule | Equation result | Table value |
| --- | ---: | ---: |
| `(1,4,8)x8x8` | 8,624 | 1,782 |
| `(2,4,8)x4x4` | 21,952 | 2,564 |
| `(2,4,8)x8x8` | 5,488 | 1,288 |
| `(2,4,8)x16x16` | 1,372 | 332 |

This is not a rounding issue. A missing spatial downsampling, token pooling, or
different crop resolution must be disclosed. Until then, computational claims
based on the table are not reproducible.

### Clip length and stride

The paper correctly notes late in the discussion that, when `F` is the number
of sampled frames, stride 2 increases source-time coverage rather than reducing
the model's input token length. For example, 32 frames at stride 2 span 63 source
frames. Accordingly, the ablation jointly changes sampling density and temporal
coverage. It does not isolate either variable.

### 2D versus 3D patches

The 3D row is stronger on every metric, which is consistent with joint local
spatio-temporal encoding. Attribution requires equal token count, parameter
count, encoder width/depth, and compute. Those controls are not stated. The table
caption calls the experiment BAUM-1, yet its full-model numbers exactly reproduce
the DAiSEE row, suggesting a dataset-caption or copied-result error.

### Global depth and width

Depth 4/width 256 has the best reported balance; depth 6 has higher precision but
lower recall. The depth-8 row also changes width to 384, so the effect of depth
cannot be separated from width. The repository adds a depth-4/width-384 control.

## State-of-the-art comparisons and claims

SEFormer's DAiSEE accuracy gain over the strongest listed baseline is 2.87
points, but the recall gain is only 0.08. BAUM recall is below the strongest
listed baseline. Without identical splits, preprocessing, class subsets,
averaging definitions, repeated-run variance, or statistical tests, these are
reported-score comparisons—not controlled head-to-head evidence. The conclusion
should say the method achieves higher reported accuracy/precision under the
authors' protocol, not that it universally or consistently outperforms all
alternatives.

## What the repository resolves—and what it cannot

The code resolves ambiguity through explicit configuration, fixed class orders,
subject-leakage checks, deterministic seeds, saved environments/configs,
paper-compatible and standard F1 definitions, computed token/parameter audits,
per-sample predictions, and all stated ablation switches. It cannot recover
unpublished splits, hyperparameters, trained weights, or experimental randomness.
Those must come from the authors before the manuscript's exact scores can be
claimed as reproduced.
