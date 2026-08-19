# Ablation configuration map

Every YAML inherits the paper-aligned DAiSEE configuration and replaces only
the fields named by the experiment. No reported score is stored in a config.

| Manuscript experiment | Configuration(s) |
| --- | --- |
| Full SEFormer | `../daisee.yaml` |
| w/o CVAF (late fusion) | `no_cvaf.yaml` |
| w/o global encoder (MLP) | `no_global_encoder.yaml` |
| w/o sequence pooling (CLS) | `cls_pooling.yaml` |
| Single-view mid | `single_view_mid.yaml` |
| Two views, coarse + fine | `two_views_coarse_fine.yaml` |
| Approximately parameter-matched views | `parameter_matched_{1,2,3}views.yaml` |
| Temporal-dense `(1,4,8)x8x8` | `patch_temporal_dense.yaml` |
| Spatial-fine `(2,4,8)x4x4` | `patch_spatial_fine.yaml` |
| Spatial-coarse `(2,4,8)x16x16` | `patch_spatial_coarse.yaml` |
| 16/32/64 frames and stride 1/2 | `clip*_stride*.yaml` plus `../daisee.yaml` |
| 2D patches + temporal pooling | `patchification_2d.yaml` |
| Global depth 0/2/4/6/8 | `global_depth*.yaml` plus `../daisee.yaml` |

Additional controls address confounds in the manuscript:

- `global_depth4_width384_control.yaml` isolates width at the selected depth;
- `cvaf_bidirectional_control.yaml` tests the abstract's bidirectional claim;
- `no_vertical_flip_control.yaml` tests whether upside-down face augmentation
  is helpful or harmful.

The budgeted view widths (624 for one view, 400 for two, and 320 for three) are
repository assumptions selected analytically to produce approximately 14.2M,
14.1M, and 14.3M parameters. They are not claimed to be the unpublished widths
used for the manuscript's 14.2M/14.5M/14.9M rows. Run `scripts/benchmark.py`
for the exact instantiated count in the installed PyTorch environment.
