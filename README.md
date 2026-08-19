# SEFormer

Code implementation for **“SEFormer: Cross-View Spatio-Temporal
Transformer for Student Engagement Recognition from Facial Video Cues.”**

SEFormer represents a face video at three temporal granularities, encodes each
view independently, transfers information from finer to coarser token streams
with cross-view attention fusion (CVAF), pools every view with learned token
weights, and integrates the pooled view tokens with a global Transformer.


## Architecture

For an input tensor `B x C x F x H x W`, the default paper-aligned model uses:

1. MTCNN face localization and resize to `112 x 112`.
2. Three non-overlapping Conv3D tubelet projections with kernels
   `(2,8,8)`, `(4,8,8)`, and `(8,8,8)`.
3. A dedicated pre-norm Transformer encoder for each view.
4. Adjacent-view CVAF, ordered by token count, with the coarser stream as query
   and the next finer stream as key/value.
5. Attention-based sequence pooling over each encoded stream.
6. Projection of pooled views to a common width, a four-layer global
   Transformer, global sequence pooling, and an MLP classifier.


## Installation

Python 3.10--3.12 and a CUDA-capable PyTorch installation are recommended. The
dependency bounds intentionally match the current `facenet-pytorch` MTCNN
release (`torch 2.2.x`, `torchvision 0.17.x`).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[face,dev]'
```

For a containerized environment:

```bash
docker build -t seformer .
```

## Data access 


- **DAiSEE:** request the approximately 15 GB archive from the
  [official dataset page](https://people.iith.ac.in/vineethnb/resources/daisee/index.html)
  and accept its terms. Keep the official Train/Validation/Test split.
- **BAUM-1s:** follow the access instructions in the
  [UCI record](https://archive.ics.uci.edu/dataset/473/baum%2B1). The paper does
  not specify a split; provide the authors' split in the annotation CSV for an
  exact comparison. Otherwise the preparation script creates a deterministic,
  subject-disjoint exploratory split and labels it as non-paper-exact.

All training code consumes a CSV manifest with these columns:

| Column | Required | Meaning |
| --- | --- | --- |
| `path` | yes | Absolute or manifest-relative video/NPZ path |
| `label` | yes | Zero-based integer class index |
| `split` | yes | `train`, `val`, or `test` |
| `sample_id` | yes | Stable unique clip identifier |
| `subject_id` | recommended | Participant identifier for leakage checks |
| `label_name` | recommended | Human-readable class name |

Prepare and validate DAiSEE:

```bash
python scripts/prepare_daisee.py \
  --root /data/DAiSEE \
  --output data/daisee_manifest.csv
python scripts/validate_manifest.py \
  --manifest data/daisee_manifest.csv \
  --num-classes 4
```

Prepare BAUM-1s from a local annotation table:

```bash
python scripts/prepare_baum1s.py \
  --root /data/BAUM-1s \
  --annotations /data/BAUM-1s/annotations.csv \
  --path-column path --label-column emotion \
  --subject-column subject_id --split-column split \
  --output data/baum1s_manifest.csv
```

Precompute face crops. Output is compressed NPZ and a replacement
manifest.

```bash
python scripts/preprocess_faces.py \
  --manifest data/daisee_manifest.csv \
  --output-dir data/daisee_faces \
  --output-manifest data/daisee_faces.csv \
  --device cuda
```

On detection failure, the preprocessor reuses the last valid box; before the
first detection it uses a centered square crop. It records a detection rate for
every clip so failures can be audited.

## Training

Update `data.manifest` in the dataset config or override it on the command line:

```bash
python scripts/train.py \
  --config configs/daisee.yaml \
  --set data.manifest=data/daisee_faces.csv \
  --set output.dir=runs/daisee_seed42
```

The default objective is multiclass cross-entropy with label smoothing. AdamW,
cosine learning-rate decay, stochastic depth, Gaussian noise, and vertical flip
are implemented.

Evaluate the selected checkpoint and save predictions, macro metrics,
per-class metrics, and confusion matrix:

```bash
python scripts/evaluate.py \
  --config runs/daisee_seed42/config.resolved.yaml \
  --checkpoint runs/daisee_seed42/best.pt \
  --split test
```

Predict one raw video:

```bash
python scripts/predict.py \
  --config runs/daisee_seed42/config.resolved.yaml \
  --checkpoint runs/daisee_seed42/best.pt \
  --video /path/to/clip.avi \
  --face-detection
```

## Ablations and audits

Run one ablation:

```bash
python scripts/train.py --config configs/ablations/no_cvaf.yaml \
  --set data.manifest=data/daisee_faces.csv
```

Run a selected configuration matrix serially:

```bash
python scripts/run_ablations.py \
  --configs 'configs/ablations/*.yaml' \
  --set data.manifest=data/daisee_faces.csv
python scripts/aggregate_runs.py \
  --runs 'runs/ablations/*' \
  --output runs/ablation_summary.csv
```

## License

The code is released under the MIT License. Dataset licenses and access terms
are separate and continue to apply.
