
# 10/24/2020

import re
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
import pandas as pd
import torchvision
verbose = False


class EchoECG(torch.utils.data.Dataset):
    
    def __init__(self, root=None,split="Train",target_type="EF",target_transform=None,batch_size = 1,csv = None):
        if root is None:
            root = "EchoNet_ECG_waveforms"
        if csv is None:
            csv =  "EFFileList.csv"
        self.folder = pathlib.Path(root)
        self.split = split.upper()
        self.target_type = target_type
        self.target_transform = target_transform
        self.file_list = pd.read_csv(csv)
        
        self.file_list = self.file_list[self.file_list.Split==self.split].reset_index()
        self.batch_size = batch_size
    def __getitem__(self, index):
        #if index>=len(self.file_list.index):
        #    raise IndexError()
        # Find filename of video
        if index >= len(self):
            raise IndexError()
        fname = self.file_list.Filename[index%len(self.file_list.index)][:-4]+'.npy'
        waveform = np.load(os.path.join(self.folder, fname))
        if self.target_type == 'DISCHARGE_DISPOSITION':
            if self.file_list[self.target_type][index]=='EXPIRED':
                target = 1#Tensor([1])
            else:
                target = 0#Tensor([0])
        
        elif self.target_type == 'ICU_ADMIT_TIME':
            target = not pd.isnull(self.file_list[self.target_type][index])
        else:
            target = self.file_list[self.target_type][index]
        if not self.target_transform is None:
            for i in self.target_transform:
                if i == "Leads":
                    point = np.random.randint(0,12)
                    waveform = np.roll(waveform,point,axis = 1)
                if i == "Start":
                    point = np.random.randint(0,5000)
                    waveform = np.roll(waveform,point)
    
        waveform = Tensor([waveform])
        return waveform, target#self.file_list[self.target_type][index]

    def __len__(self):

        return len(self.file_list.index)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.
    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)
