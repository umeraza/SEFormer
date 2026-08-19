# Primary sources checked

Accessed 19 August 2026.

- [DAiSEE official page](https://people.iith.ac.in/vineethnb/resources/daisee/index.html):
  9,068 clips, 112 users, four affect dimensions, four intensity levels, access
  form, and research-use/reproduction conditions.
- [DAiSEE paper](https://arxiv.org/abs/1609.01885): dataset design and benchmark
  framing.
- [BAUM-1 official UCI record](https://archive.ics.uci.edu/dataset/473/baum%2B1):
  1,184 clips, 31 subjects, 13 emotional/mental states, access instructions,
  DOI, and CC BY 4.0 record metadata.
- [EngageFormer paper](https://arxiv.org/abs/2502.10813): closest architectural
  predecessor and source checked for its reported 512/1024 view encoder,
  training duration, learning rate, weight decay, tubelets, and augmentations.
- [facenet-pytorch repository](https://github.com/timesler/facenet-pytorch):
  MTCNN detection/tracking API and dependency compatibility.
- [PyTorch automatic mixed precision](https://docs.pytorch.org/docs/stable/amp.html),
  [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html),
  and [cosine annealing](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html):
  execution semantics used in the training engine.
- [scikit-learn classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
  and [precision/recall/F-score support](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html):
  fixed-label per-class and macro averaging semantics.

The supplied SEFormer manuscript remains the authority for the proposed method
and its claimed results. External sources were used to verify public dataset and
software facts and to identify where the manuscript inherits or omits details;
they were not used to fabricate unpublished SEFormer settings.
