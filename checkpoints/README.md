# checkpoints

Trained predictor checkpoints. Too large for git — hosted on Google Drive.

Download from: https://drive.google.com/... *(replace with your share link)*

Place these folders here:

- `opt125m-ltr-original/` — OPT-125M predictor trained with the original paper's
  ListMLE loss (`--loss_type listmle`)
- `opt125m-ltr-marginloss/` — OPT-125M predictor trained with this project's
  pairwise margin-ranking loss extension (`--loss_type margin`)

Produced by `scripts/0b_train_predictor.py`.
