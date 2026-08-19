# Repository structure

```text
SEFormer/
├── .github/workflows/ci.yml       # CPU lint, tests, and canonical paper audit
├── configs/
│   ├── base.yaml                  # Every model/data/training field
│   ├── daisee.yaml                # Four-level engagement task
│   ├── baum1s.yaml                # Six-basic-emotion task
│   ├── smoke.yaml                 # Tiny synthetic CPU model
│   └── ablations/                 # 24 paper ablations and controls
├── scripts/
│   ├── prepare_daisee.py          # Official-label to manifest conversion
│   ├── prepare_baum1s.py          # Annotation-driven six-class conversion
│   ├── validate_manifest.py       # Existence, label, duplicate, leakage checks
│   ├── preprocess_faces.py        # MTCNN crops and detection-quality metadata
│   ├── train.py                   # AdamW/cosine training and validation selection
│   ├── evaluate.py                # Explicit split evaluation and artifacts
│   ├── predict.py                 # One-video prediction and attention summary
│   ├── run_ablations.py           # Isolated serial experiment matrix
│   ├── aggregate_runs.py           # Comparable run/parameter summary CSV
│   ├── benchmark.py               # Token, parameter, latency, and memory report
│   └── audit_paper.py             # Reproducibility and manuscript consistency audit
├── src/seformer/
│   ├── config.py                  # YAML inheritance, overrides, validation
│   ├── analysis.py                # Pure-Python token/parameter accounting
│   ├── sampling.py                # Clip indices and temporal coverage
│   ├── data.py                    # Video/NPZ decoding, transforms, loaders
│   ├── manifest.py                # Framework-independent manifest checks
│   ├── faces.py                   # MTCNN crop selection and fallbacks
│   ├── layers.py                  # Tubelets, Transformer, CVAF, sequence pooling
│   ├── model.py                   # End-to-end SEFormer
│   ├── engine.py                  # Mixed-precision train/evaluation loops
│   ├── metrics.py                 # Macro/class metrics and confusion artifacts
│   ├── checkpoint.py              # Versioned atomic checkpoints
│   └── utils.py                   # Seeds, device, environment, serialization
├── tests/                         # Formula, config, metric, shape, gradient tests
├── README.md                      # End-to-end usage
├── REPRODUCIBILITY.md             # Specified vs inferred values and contradictions
├── PAPER_ANALYSIS.md              # Critical method/data/metric/ablation review
├── MODEL_CARD.md                  # Intended use, privacy, bias, limitations
├── SOURCES.md                     # Primary dataset/paper/software references
├── CITATION.cff                   # GitHub citation metadata
├── LICENSE                        # Code license; datasets remain separate
├── pyproject.toml                 # Installable package and dependency bounds
├── requirements.txt               # Pip-compatible locked compatibility ranges
├── environment.yml                # CUDA/Conda environment
├── Dockerfile                     # Reproducible CUDA runtime
├── Makefile                       # Common checks
└── .pre-commit-config.yaml         # Ruff lint/format hooks
```

The repository intentionally excludes raw datasets, derived facial crops,
checkpoints, and result claims. Those are large, privacy-sensitive, governed by
separate terms, or must be regenerated. `.gitignore` blocks common media and
weight formats to reduce accidental publication risk.
