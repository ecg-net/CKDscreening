
# 10/24/2020

import os, os.path
from os.path import splitext
import numpy as np
import torch
from torch import Tensor
import shutil
from multiprocessing import dummy as multiprocessing
import time
import subprocess
import datetime
from datetime import date
import sys
import cv2
import matplotlib.pyplot as plt
import sys
from shutil import copy
import math
import torch
import torchvision
import torch.utils.data
import torch.nn as nn
import skimage.draw
import pathlib
import collections
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
import pandas as pd
import torchvision
verbose = False


def rolling_average (array,value_length):
    new_array = np.zeros((1,5000,12))
    assert array.shape == (1,5000,12), "array is not shape (1,2500,12)"
    for i in range(0,12):
        new_array[0,:,i]=pd.Series(array[0][:,i]).rolling(window=value_length,min_periods=1).mean() #min_periods ensure no NaNs before value_length fulfilled
    return new_array


# +
# 2, 4 , 8, 16, 
# -

5000//64


def plot(array,color = 'blue'):
    lead_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    plt.rcParams["figure.figsize"] = [16,9]
    
    fig, axs = plt.subplots(len(lead_order))
    fig.suptitle("array")
    # rolling_arr = rolling_average(array,15)
    if array.shape == (5000, 12):
        for i in range(0,12):
            axs[i].plot(array[:2500,i],label = 'window')
            # axs[i].plot(array[::2,i],label = 'downsample')
            # axs[i].plot(rolling_arr[:2500,i],label = 'rolling')
            axs[i].set(ylabel=str(lead_order[i]))
    elif array.shape == (12, 5000):
        for i in range(0,12):
            axs[i].plot(array[i,:2500],label = 'window')
            # axs[i].plot(array[i,::2],label = 'downsample')
            # axs[i].plot(rolling_arr[i,:],label = 'rolling')
            axs[i].set(ylabel=str(lead_order[i]))
    elif array.shape == (1,5000,12):
        for i in range(0,12):
            axs[i].plot(array[0,:5000,i],label = 'window')
            # axs[i].plot(array[0,::2,i],label = 'downsample')
            # axs[i].plot(rolling_arr[0,:2500,i],label = 'rolling')
            axs[i].set(ylabel=str(lead_order[i]))
    elif array.shape == (1,1,5000,12):
        for i in range(0,12):
            axs[i].plot(array[0,0,:5000,i],label = 'window')
            # axs[i].plot(array[0,0,::2,i],label = 'downsample')
            # axs[i].plot(rolling_arr[0,0,:2500,i],label = 'rolling')
            axs[i].set(ylabel=str(lead_order[i]))
    else:
        print("ECG shape not valid: ",array.shape)
    
    plt.show()


class EchoECG(torch.utils.data.Dataset):
    
    def __init__(self, root=None,csv = None,bootstrap = False,sliding = False,downsample=0,model = 'Columbia'
                 ,rolling = 0,plot_data = False,additional_inputs = None
                 ,target = 'Phenocode',Filename = 'Filename',
                return_filename = False, one_lead=False):
        if root is None:
            root = "EchoNet_ECG_waveforms"
        if csv is None:
            csv =  "EFFileList.csv"
        self.folder = pathlib.Path(root)
        df = pd.read_csv(csv)
        self.Filename = Filename
        print('missing',1-sum(~pd.isna(df[self.Filename]))/len(df))
        df = df[~pd.isna(df[self.Filename])].reset_index(drop=True)
        print(df.columns)
        if bootstrap:
            self.file_list = self.file_list.sample(frac=0.5, replace=True).reset_index(drop=True)
        
        if target == 'Phenocode':
            self.data = df[df.columns[1:-2]].to_numpy()
        elif target == 'Race':
            self.data = df[['is_Black', 'is_White', 'is_Asian','is_Other']].to_numpy()
        else:
            df = df[~pd.isna(df[target])].reset_index(drop=True)
            self.data = df[[target]].to_numpy()
        self.file_list = df[self.Filename]
        self.sliding = sliding
        self.downsample = downsample
        self.model = model
        self.rolling = rolling
        self.plot_data = plot_data
        self.additional_inputs = additional_inputs
        if not self.additional_inputs is None:
            self.extra_inputs_list = df[additional_inputs]
        self.target = target
        self.return_filename = return_filename
        self.one_lead = one_lead
        
    def __getitem__(self, index):
        fname = self.file_list[index%len(self.file_list)]
    
        try:
            waveform = np.load(os.path.join(self.folder, fname))
        except FileNotFoundError:
            print('missing')
            return(None)
        
        if waveform.shape[0]==5000:
            waveform = waveform.T
        waveform = np.expand_dims(waveform,axis=0)
        
        x = []
        if not self.additional_inputs is None:
            for i in self.additional_inputs:
                x.append(self.extra_inputs_list[i][index%len(self.file_list.index)])
        start = np.random.randint(2499)
        if self.rolling != 0:
            waveform = rolling_average(waveform,self.rolling)
        if self.plot_data == True:
            plot(waveform,color = 'orange')
            plt.show()
        if self.sliding:
            waveform = waveform[:,start:start+2500]
        if self.downsample>0:
            waveform = waveform[:,::self.downsample,:]
        if self.model == 'eff2d':
            waveform = np.array([waveform[0,:,:],waveform[0,:,:],waveform[0,:,:]])
            waveform = np.reshape(waveform,(3,150,200))
        if not self.model in ['Columbia', 'eff2d','Old Columbia', 'Wacky']:
            waveform = waveform[0,:,:]
        if self.model =='efficency':
            waveform = torch.FloatTensor(waveform)
        else:
            waveform = torch.FloatTensor(waveform)
        if self.target == 'Phenocode':
            target = torch.FloatTensor(self.data[index,:])
        else:
            target = torch.FloatTensor(self.data[index])
            
        if self.one_lead:
            waveform = waveform[0:1]
            
        if self.return_filename:
            if not self.additional_inputs is None:
                return (waveform,torch.FloatTensor(x)), target,fname
            else:
                return waveform, target,fname
        else:
            
            if not self.additional_inputs is None:
                return (waveform,torch.FloatTensor(x)), target
            else:
                return waveform, target
    

    def __len__(self):

        return math.ceil(len(self.file_list.index))


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.
    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


