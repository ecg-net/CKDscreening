# ckd-cross-validation
Testing repo to allow for external validation of Cedars-Sinai research into using deep learning to predict chronic kidney disease

## Inclusion criteria
* EKGs are paired with ICD9 codes 585.1-6 for CKD stages I-VI respectively within one year of the date of the EKG.
* EKGs are paired with the closest-in-time diagnosis code that is within the window.
* Each EKG is only paired with one diagnosis record.
* Negative examples are only taken from patients who have never had any instances of the diagnosis codes on their record.

## EKG Preprocessing
* Data obtained from MUSE dataset, expected dimensions are 12x5000 at 500Hz
* Examples were run through dataset-level normalization (mean 0, std 1)
* For one-lead model testing, model inputs remain the same shape, the dataloader handles using only one channel.

## Testing task
* Dataloader expects a CSV containing the following columns:
    * "Filename", containing the .npy filename of the EKGs
    * "label", containing a 1 or 0 indicating whether or not the patient had CKD within a year of the EKG
    * "age", age of patient in years at time of EKG.
* Testing notebook has variables at the top for a root path to the folder containing all .npy files as well as one for the path to the testing CSV.
* Model outputs a prediction between 0 and 1 representing whether or not it predicts the patient has the diagnosis within the window.
* Testing speed should be in the ballpark of ~800 examples per second
* Model weights are already in repo and referenced relatively.

## Testing Procedure
* Install requirements.txt
* Open TestingClassifier.ipynb, edit and run all cells. AUC values will be printed at the bottom of the notebook.
