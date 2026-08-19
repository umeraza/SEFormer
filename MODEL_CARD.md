# SEFormer model card

## Intended use

SEFormer is research software for clip-level classification of observable facial
video cues. The supplied configurations cover four ordinal DAiSEE engagement
intensities and six BAUM-1s basic-emotion classes.

It may support offline benchmarking, ablation research, and investigation of
multi-scale video Transformers. It is not a validated instrument for measuring
a learner's cognitive state, motivation, aptitude, attention, or academic value.

## Out-of-scope use

Do not use this model for disciplinary action, grading, admissions, employment,
health assessment, surveillance, or other high-impact decisions. Do not infer a
person's internal state from a single prediction. Webcam-derived facial signals
are indirect, culturally and individually variable, and affected by pose,
lighting, disability, camera quality, occlusion, and annotation subjectivity.

## Inputs and outputs

- Input: a face-centered RGB video clip, normally 32 sampled frames resized to
  112x112.
- Output: uncalibrated logits and softmax probabilities over a configured class
  order.
- Optional diagnostics: per-view and global sequence-pooling weights. Attention
  weights are not causal explanations.

## Data

DAiSEE and BAUM-1s are third-party datasets with their own access conditions.
This repository contains neither. Dataset documentation and consent constraints
must be reviewed before use. Never commit raw videos, crops, or identifying
metadata to a public repository.

## Known limitations

- Face-only input omits posture, interaction context, audio, and learning logs.
- Clip labels can hide within-clip state changes.
- DAiSEE is strongly imbalanced; very-low engagement has very limited test
  support in the manuscript.
- The manuscript reports substantial BAUM confusion for disgust/surprise and
  anger, and its split protocol is unspecified.
- Cross-dataset results do not establish cross-domain generalization.
- The architecture can be expensive: full self-attention on the finest view is
  quadratic in its 3,136 patch tokens for the stated default input.
- Several training hyperparameters in the repository are inferred, not reported.

## Evaluation recommendations

Report raw accuracy, macro precision/recall/F1, balanced accuracy, per-class
support, raw and normalized confusion matrices, multiple seeds, uncertainty
intervals, and subject-disjoint generalization. Audit performance by relevant
demographic and acquisition groups when legally and ethically appropriate.
Calibrate probabilities before any thresholded downstream use.

## Privacy and security

Facial video is sensitive biometric-adjacent data. Minimize retention, restrict
access, encrypt storage and transport, document lawful basis and consent, and
delete derived crops when no longer needed. Checkpoints can retain information
about training data; do not publish them without a release review.
