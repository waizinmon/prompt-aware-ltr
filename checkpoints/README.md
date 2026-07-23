# checkpoints

Trained predictor checkpoints. Too large for git — hosted on Google Drive.

Download from: https://fduedu-my.sharepoint.com/:f:/g/personal/w_mon_student_fdu_edu/IgAvHVaOSnjZTIrRuoEavoI7AR16ppt9W9SzO8H9SN7xspY?e=jdQ3AC

Place these folders here:

- `opt125m-ltr-original/` — OPT-125M predictor trained with the original paper's
  ListMLE loss (`--loss_type listmle`)
- `opt125m-ltr-marginloss/` — OPT-125M predictor trained with this project's
  pairwise margin-ranking loss extension (`--loss_type margin`)

Produced by `scripts/0b_train_predictor.py`.
