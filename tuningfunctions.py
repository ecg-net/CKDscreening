# +
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import Resize, ToTensor, Normalize
import matplotlib.pyplot as plt
# from imblearn.under_sampling import RandomUnderSampler
import cv2
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, \
    average_precision_score
from sklearn.model_selection import train_test_split
import time

from importlib import reload
import os
from pathlib import Path
from skimage import io
import copy
from torch import optim, cuda
import pandas as pd
import glob
from importlib import reload
from collections import Counter
# Useful for examining network
from functools import reduce
from operator import __add__
# from torchsummary import summary
import seaborn as sns
import warnings
# warnings.filterwarnings('ignore', category=FutureWarning)
from PIL import Image
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import wandb
# -

# Useful for examining network
from functools import reduce
from operator import __add__
from torchsummary import summary

# from IPython.core.interactiveshell import InteractiveShell
import seaborn as sns

import warnings
# warnings.filterwarnings('ignore', category=FutureWarning)

# Image manipulations
from PIL import Image

# Timing utility
from timeit import default_timer as timer

# Visualizations
import matplotlib.pyplot as plt




import optuna
from ignite.engine import Engine
from ignite.engine import create_supervised_evaluator
from ignite.engine import create_supervised_trainer
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss, Precision, Recall, Fbeta
from ignite.contrib.metrics.roc_auc import ROC_AUC
from ignite.contrib.metrics.regression import R2Score
from ignite.handlers import ModelCheckpoint, global_step_from_engine, Checkpoint, DiskSaver
from ignite.handlers.early_stopping import EarlyStopping
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers import ProgressBar


import models

"""
val_metrics = {
        "accuracy": Accuracy(output_transform=thresholded_output_transform),
        "loss": Loss(self.criterion),
        "roc_auc": ROC_AUC(output_transform=thresholded_output_transform),
        "precision": Precision(output_transform=thresholded_output_transform),
        "precision_0": Precision(output_transform=class0_thresholded_output_transform),
        "recall": Recall(output_transform=thresholded_output_transform),
        "recall_0": Recall(output_transform=class0_thresholded_output_transform),
        }
    val_metrics["f1"]=Fbeta(beta=1.0, average=False, precision=val_metrics['precision'], recall=val_metrics['recall'])
    val_metrics["f1_0"]=Fbeta(beta=1.0, average=False, precision=val_metrics['precision_0'], recall=val_metrics['recall_0'])

                """


def get_data_loaders(X_train, X_test, y_train, y_test):
  
    batch_size = 20
    dlen = X_train.shape[0]


    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    X_test = TensorDataset(torch.FloatTensor(X_test), y_test)
    test_loader = DataLoader(X_test, batch_size=batch_size, pin_memory=True, shuffle=True,num_workers = 10)

    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_train = TensorDataset(torch.FloatTensor(X_train), y_train)
    train_loader = DataLoader(X_train, batch_size=batch_size, pin_memory=True, shuffle=True,num_workers = 10)

    return train_loader, test_loader


def get_criterion(y_train):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Train on: {device}')


    LABEL_WEIGHTS = []

   
    class_counts = np.bincount(y_train).tolist() #y_train.value_counts().tolist()
    weights = torch.tensor(np.array(class_counts) / sum(class_counts))
    # assert weights[0] > weights[1]
    print("CLASS 0: {}, CLASS 1: {}".format(weights[0], weights[1]))
    weights = weights[0] / weights
    print("WEIGHT 0: {}, WEIGHT 1: {}".format(weights[0], weights[1]))
    LABEL_WEIGHTS.append(weights[1])

    print("Label Weights: ", LABEL_WEIGHTS)
    cuda_idx = 0
    LABEL_WEIGHTS = torch.stack(LABEL_WEIGHTS)
    LABEL_WEIGHTS = LABEL_WEIGHTS.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=LABEL_WEIGHTS)
    criterion.to(device)
    
    return criterion

def thresholded_output_transform(output):
            y_pred, y = output
            y_pred = torch.round(torch.sigmoid(y_pred))
            return y_pred, y
def class0_thresholded_output_transform(output):
            y_pred, y = output
            y_pred = torch.round(torch.sigmoid(y_pred))
            y=1-y
            y_pred=1-y_pred
            return y_pred, y
def class0_thresholded_output_transform_mayo(output):
            y_pred, y = output
            _, y_pred = torch.max(y_pred.data, 1)
            y=1-y
            y_pred=1-y_pred
            return y_pred, y
def thresholded_output_transform_mayo(output):
            y_pred, y = output
            _, y_pred = torch.max(y_pred.data, 1)
            return y_pred, y


# !pwd

class Objective(object):
    def __init__(self, model_name, criterion, train_loader, test_loader, optimizers, lr_lower, 
                 lr_upper, metric, max_epochs, early_stopping_patience=None, lr_scheduler=False, 
                 step_size=None, gamma=None,Project_name = None,model_origin = None,reduce = False,
                 depth = [1,2,2,3,3,3,3],channels = [32,16,24,40,80,112,192,320,1280,1867],
                 dilation = 1,stride = 2,expansion = 6,additional_inputs = 0,category = 'roc',
                 multi=False,reg = False, start_channels=12, multi_class=False):
        # Hold this implementation specific arguments as the fields of the class.
        self.model_name=model_name
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.optimizers = optimizers
        self.criterion=criterion
        self.metric = metric
        self.max_epochs=max_epochs
        self.lr_lower=lr_lower
        self.lr_upper=lr_upper
        self.early_stopping_patience=early_stopping_patience
        self.lr_scheduler=lr_scheduler
        self.step_size=step_size
        self.gamma=gamma
        self.Project_name = Project_name
        self.model_origin = model_origin
        self.reduce = reduce
        self.depth = depth
        self.channels = channels
        self.start_channels = start_channels
        self.dilation = dilation
        self.stride = stride
        self.exp = expansion
        self.additional_inputs = additional_inputs
        self.type = category
        if self.type == 'regression':
            self.metric = 'loss'
        print('self.exp',self.exp)
        self.multi_class=multi_class
        self.multi=multi
        self.reg = reg
    def __call__(self, trial):
        torch.cuda.empty_cache()
        # load model
        model = None
        if self.model_origin == 'Columbia':
            model = getattr(models, self.model_name)(inital_kernel_num = 32,dropout = 0.6,conv1kernel1 = 7,conv1kernel2 = 3,reduced = self.reduce)
        elif self.model_origin == 'Lancet':
            model_type='lancet'
            is_2d=False
            class_name = 'hemoglobin'
            lr = 1e-3
            model_name = 'lancet'
            binary=False
            batch_norm=True
            drop_prob=0.
            conv_width=3
            train_delta=0
            first_layer_out_channels=None
            first_layer_kernel_size=None
            normalize_x="lead"
            binary_cutoff = None
            normalize_y=False
            from ecgnet import model 
            reload(model)
            model = model.build_model(model_type, 1, 12 if is_2d else 12, is_2d=is_2d, binary=binary, batch_norm=batch_norm, default_conv_kernel=conv_width, drop_prob=drop_prob,
                                     binary_cutoff=binary_cutoff, first_layer_out_channels=first_layer_out_channels, first_layer_kernel_size=first_layer_kernel_size)
        elif self.model_origin == 'mayo':
            model = models.Mayo_Net()
        elif self.model_origin == 'eff2d':
            
            model=model.EfficientNet.from_name('efficientnet-b0')
            model._fc = nn.Linear(1280,1)
        elif self.model_origin == 'RCRI_Net':
            model = models.EffNet(depth = self.depth,channels = self.channels,dilation = self.dilation,
                                  stride = self.stride,expansion = self.exp, num_additional_features=self.additional_inputs,
                                 multi_class = self.multi_class, reg = self.type=='regression', start_channels=self.start_channels)
            print(model)
            print('model: ' +str(self.depth)+" channels: "+str(self.channels)+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        if self.model_origin == 'Old Columbia':
            import old_models
            model = getattr(old_models, self.model_name)(inital_kernel_num = 32,dropout = 0.6,conv1kernel1 = 7,conv1kernel2 = 3)
        if self.model_origin == 'Wacky':
            import old_models
            model = getattr(old_models, 'Wacky')(inital_kernel_num = 32,dropout = 0.6,conv1kernel1 = 7,conv1kernel2 = 3)
            print(model)
            print('model: ' +str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
            
        wandb.watch(model)

        # print(model)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            model.to(device)
            model = nn.DataParallel(model)
        if self.type == 'regression':
            val_metrics = {
            "loss": Loss(self.criterion),
            "r2": R2Score()}
        else:
            val_metrics = {
                "loss": Loss(self.criterion),
                "roc_auc": ROC_AUC()
                }
            


        optimizer_name = trial.suggest_categorical("optimizer", self.optimizers)
        learnrate = trial.suggest_loguniform("lr", self.lr_lower, self.lr_upper)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learnrate)
        trainer = create_supervised_trainer(model, optimizer, self.criterion, device=device)
        ProgressBar().attach(trainer)
        train_evaluator = create_supervised_evaluator(model, metrics= val_metrics, device=torch.device('cuda'))
        evaluator = create_supervised_evaluator(model, metrics= val_metrics, device=torch.device('cuda'))

        # Register a pruning handler to the evaluator.
        pruning_handler = optuna.integration.PyTorchIgnitePruningHandler(trial, self.metric, trainer)
        evaluator.add_event_handler(Events.COMPLETED, pruning_handler)
        if not self.type == 'regression':
            def score_fn(engine):
                if self.type == 'regression':
                    score = engine.state.metrics['loss']
                    return score
                else:
                    score = engine.state.metrics[self.metric]
                return score if self.metric!='loss' else -score
        def score_fn_2(engine):
            score = engine.state.metrics['loss']
            return -score

        #early stopping
        if not self.type == 'regression':
            if self.early_stopping_patience is not None:
                es_handler = EarlyStopping(patience=self.early_stopping_patience, score_function=score_fn, trainer=trainer)
                evaluator.add_event_handler(Events.COMPLETED, es_handler)
        else:
            if self.early_stopping_patience is not None:
                es_handler = EarlyStopping(patience=self.early_stopping_patience, score_function=score_fn_2, trainer=trainer)
                evaluator.add_event_handler(Events.COMPLETED, es_handler)

        #checkpointing
        to_save = {'model': model}

        checkpointname=self.Project_name+'/checkpoint'
        for key, value in trial.params.items():
          checkpointname+=key+': '+str(value)+', '
        if not self.type == 'regression':
            checkpoint_handler = Checkpoint(to_save, DiskSaver(checkpointname, create_dir=True,require_empty=False),
                             filename_prefix='best_roc', score_function=score_fn, score_name="val_roc",
                             global_step_transform=global_step_from_engine(trainer))
        else:
            checkpoint_handler = Checkpoint(to_save, DiskSaver(checkpointname, create_dir=True,require_empty=False),
                             filename_prefix='best_roc', score_function=score_fn_2, score_name="val_roc",
                             global_step_transform=global_step_from_engine(trainer))
        checkpoint_handler_2 = Checkpoint(to_save, DiskSaver(checkpointname, create_dir=True,require_empty=False),
                         filename_prefix='best_loss', score_function=score_fn_2, score_name="val_loss",
                         global_step_transform=global_step_from_engine(trainer))
        wandb.log({'Checkpoint Name':checkpointname},commit=False)
        print(checkpointname)
        evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler)
        evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler_2)

        #  Add lr scheduler
        if self.lr_scheduler is True:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
            trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: scheduler.step())
            print(scheduler)

        
        
        #print metrics on each epoch completed
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            train_evaluator.run(self.train_loader)
            metrics = train_evaluator.state.metrics
            if self.type == 'regression':
                wandb.log({'Train loss':metrics["loss"],
                          'Train r2':metrics['r2']},commit=False)
            else:
                wandb.log({'Train loss':metrics["loss"],'Train roc_auc':metrics['roc_auc']},commit=False)
          

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(self.test_loader)
            metrics = evaluator.state.metrics
            if self.type == 'regression':
                wandb.log({'Val loss':metrics["loss"],
                          'Val r2':metrics["r2"]})
            else:
                wandb.log({'Val loss':metrics["loss"],'Val roc_auc':metrics['roc_auc']})

        #Tensorboard logs


        #run the trainer
        trainer.run(self.train_loader, max_epochs=self.max_epochs)

        #load the checkpoint with the best validation metric in the trial
        to_load = to_save
        checkpoint = torch.load(checkpointname+'/'+checkpoint_handler.last_checkpoint)
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

        evaluator.run(self.test_loader)
        if self.type == 'regression':
            return evaluator.state.metrics['loss']
        return evaluator.state.metrics[self.metric]


def run_trials(objective, pruner, num_trials, direction,project_name): 
    wandb.init(project=project_name,settings=wandb.Settings(start_method='thread'))
    
    pruner = pruner
    study = optuna.create_study(direction=direction, pruner=pruner)
    study.optimize(objective, n_trials=num_trials, gc_after_trial=True,show_progress_bar=False)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
          print("    {}: {}".format(key, value))


import torch

torch.__version__


