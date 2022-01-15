root = '/path/to/npy/files/root'
test_csv = '/path/to/filename/and/labels/csv'

#name of the column in test_csv where the boolean label for CKD (True)/No CKD (False) is stored
target='label'

import numpy as np
from paths import FilePaths
from tuningfunctions import get_data_loaders, get_criterion, Objective, run_trials
import models
import torch
import pandas as pd
import matplotlib.pyplot as plt
from ECG import EchoECG
from tqdm import tqdm
import sklearn

root = '/workspace/data/drives/Local_SSD/sdd/data/Remade with New Coefficents'
test_csv = '/workspace/data/drives/Local_SSD/sdc/kidney_disease/DefinitiveAllStagesData/under_60_years_old_subset_test.csv'
target='label'
base = '/workspace/kai/ckd-cross-validation/Training/'

def test_model(one_lead=False):
    torch.cuda.empty_cache()

    if(one_lead):
        model = models.EffNet(channels = [32,16,24,40,80,112,192,320,1280,1],dilation = 2,
                                  stride = 8,
                                  reg = False, 
                                  start_channels=1)
        model.load_state_dict(torch.load(base+'one_lead_weights.pt'))
    else:
        model = models.EffNet(channels = [32,16,24,40,80,112,192,320,1280,1],dilation = 2,
                                  stride = 8,
                                  reg = False, 
                                  start_channels=12)
        model.load_state_dict(torch.load(base+'twelve_lead_weights.pt'))
    model.eval()

    test_ds = EchoECG(root=root,
                      csv=test_csv,
                      model='RCRI_Net', 
                      rolling=0, 
                      downsample=1,
                      target=target, 
                      one_lead=one_lead,
                      return_filename=False)
    
    test_dataloader = torch.utils.data.DataLoader(test_ds,
                                                batch_size=2000, 
                                                num_workers=8, #Feel free to increase this depending 
                                                               #on your machine's specs to speed up inference
                                                drop_last=False)

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for ecg, labels in tqdm(test_dataloader):

            all_preds += list(model(ecg))
            all_labels += list(labels)
            
    return all_labels, all_preds


one_labels, one_preds = test_model(one_lead=True)
fpr, tpr, thresholds = sklearn.metrics.roc_curve(one_labels, one_preds)
print('One-Lead Model')
print(sklearn.metrics.auc(fpr, tpr))

twelve_labels, twelve_preds = test_model(one_lead=False)
fpr, tpr, thresholds = sklearn.metrics.roc_curve(twelve_labels, twelve_preds)
print('Twelve-Lead Model')
print(sklearn.metrics.auc(fpr, tpr))
