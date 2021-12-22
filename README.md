# ckd-cross-validation
Testing repo to allow for external validation of Cedars-Sinai research into using deep learning to predict chronic kidney disease

## Inclusion criteria

* EKGs paired with ICD9 codes 585.1-6 for CKD stages I-VI respectively within one year of the date of the EKG.
* EKGs are paired with the closest-in-time diagnosis code that is within the window.
* Negative examples are only taken from patients who have never had any instances of the positive ICD codes on their record.

## EKG Preprocessing

* Data obtained from MUSE dataset, dimension were 12x5000 at 500Hz
* Examples were run through dataset-level normalization (mean 0, std 1)
* For one-lead model testing, model inputs remain the same shape, the dataloader handles using only one channel.

## Testing task

* Model outputs a prediction between 0 and 1 representing whether or not it predicts the patient has the diagnosis within the window.
* Testing speed should be in the ballpark of ~800 examples per second
* Model weights are already in repo and referenced relatively.
