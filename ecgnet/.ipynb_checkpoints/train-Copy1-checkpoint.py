from dataset import ECGDataset
from model import build_model
from label_funcs import label_dict
from report import *
from ECG import EchoECG

import tqdm
import os
import fnmatch
import sys

import torch

import numpy as np
import sklearn.metrics

import warnings 

warnings.filterwarnings("ignore")

cutoffs = {'hemoglobin': (10, False),
           'trop': (np.log(.0001), True),
           'BNP': (np.log(300), True),
           'A1c': (6.5, False)
          }

norm_params = {'A1c': (6.36, 1.62)}

report_fns = {'A1c': A1c_report,
              'hemoglobin': hemoglobin_report}


def train(
        class_name='hemoglobin',
        label_template='/oak/stanford/groups/euan/projects/stanfordECG/labels/{}_labels.csv',
        waveforms='/oak/stanford/groups/euan/projects/stanfordECG/waveforms/',
        save_path='output/ekg',
        output='LOS_attempt8_l2',
        lr_step_period=15,
        save_all=False,
        batch_size=128,
        lr=1e-3,
        train_signal_length=4000,
        valid_signal_length=4000,
        train_delta=0,
        first_n=None,
        binary=False,
        cache=False,
        model_type='lancet',
        is_2d=False,
        conv_width=3,
        batch_norm=True,
        drop_prob=0.,
        first_layer_out_channels=None,
        first_layer_kernel_size=None,
        normalize_x="lead",
        normalize_y=False
    ):

    test_delta=0

    if lr_step_period is None:
        lr_step_period = math.inf

    binary_cutoff, log = cutoffs[class_name]
    if normalize_y:
        mean, std = norm_params[class_name]
        binary_cutoff = (binary_cutoff - mean) / std
    else:
        mean, std = None, None
    report_fn = report_fns[class_name] if class_name in report_fns else default_report

    model_name = name_model(class_name, model_type, is_2d, lr, binary, conv_width, drop_prob, batch_norm, first_layer_out_channels,
                 first_layer_kernel_size, train_delta, train_signal_length, valid_signal_length, normalize_x, normalize_y)
    if not output:
        output = os.path.join(save_path, model_name)
    #print(output)

    # labels = label_template.format(class_name)

    os.makedirs(output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lengths = {'train': train_signal_length,
               'valid': valid_signal_length
              }

    deltas = {'train': train_delta,
              'valid': test_delta}

    #datasets = {key: 
    #                ECGDataset(
    #                    labels,
    #                    waveforms,
    #                    key,
    #                    signal_length=lengths[key],
    #                    first_n=first_n if key=='train' else first_n,
    #                    max_diff=deltas[key],
    #                    binary_cutoff=None,
    #                    cache=cache,
    #                    log=log,
    #                    y_mean=mean,
    #                    y_std=std,
    #                    normalize_x=normalize_x
    #                )
    #            for key in ['train', 'valid']}


    #dataloaders = {key:
    #                torch.utils.data.DataLoader(
    #                    datasets[key],
    #                    batch_size=batch_size,
    #                    shuffle=False,
    #                    pin_memory=True)
    #            for key in ['train', 'valid']}
    root =  "../../../data/drives/data/Muse2017-2019ECG_npy"
    csv =  "../../../data/drives/data/Muse2017-2019ECG_npy.csv"
    #root = '/Users/davidouyang/Dropbox/Echo Research/CodeBase/EchoNetECGtoEF/EchoNet_ECG_waveforms/'
    #csv = '/Users/davidouyang/Dropbox/Echo Research/CodeBase/AnonymizedFullSet/EFFileList.csv'

    bs = 20
    train_ds = EchoECG(root=root,csv=csv,split='Train',target_type='LOS')
    validation_ds = EchoECG(root=root,csv=csv,split='Val',target_type = 'LOS')

    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=bs,num_workers=100)
    val_dataloader=torch.utils.data.DataLoader(validation_ds, batch_size=bs,num_workers=100)

    dataloaders = {'train':train_dataloader, 'valid':val_dataloader}


    torch.cuda.empty_cache()
    model = build_model(model_type, 1, 1 if is_2d else 12, is_2d=is_2d, binary=binary, batch_norm=batch_norm, default_conv_kernel=conv_width, drop_prob=drop_prob,
                         binary_cutoff=binary_cutoff, first_layer_out_channels=first_layer_out_channels, first_layer_kernel_size=first_layer_kernel_size).double()

    #print(model, flush=True)
    #print(type(model))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model,device_ids=[1])

    
    model.to(device)

    optim = torch.optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=0.9,
                            weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    f = open(os.path.join(output, "log.csv"), "a")

    # Attempt to load checkpoint
    epoch_resume, best_score = try_to_load(model, optim, scheduler, output, f)

    for epoch in range(epoch_resume, lr_step_period*3):
        for split in ['train', 'valid']:                
            losses, ys, yhs, score = run_epoch(split, dataloaders[split], model, optim, epoch, binary, f, device, cutoff=binary_cutoff)
            #print(ys,yhs)
        scheduler.step()

        best_score = save_model(losses, epoch, model, optim, scheduler, output, score, best_score, save_all)

    best_epoch, best_score = try_to_load(model, optim, scheduler, output, f, 'best.pt')
    print(best_score,lr_step_period)
    assert best_score > 0 or lr_step_period == 0
    
    f.write("Best epoch: {}\nBest score: {}\n".format(best_epoch, best_score))

    try:
        ys = np.load(os.path.join(output, 'y_valid.npy'))
        yhs = np.load(os.path.join(output, 'yh_valid.npy'))
    except:
        _, ys, yhs, _ = run_epoch('valid', dataloaders['valid'], model, optim, 0, binary, f, device, cutoff=binary_cutoff, log=False)
        np.save(os.path.join(output, 'y_valid.npy'), ys)
        np.save(os.path.join(output, 'yh_valid.npy'), yhs)

    report_fn(ys, yhs, model_name, [output, os.path.join(save_path, 'reports')], best_epoch, 'valid', [binary_cutoff], mean, std)



def name_model(class_name, model_type, is_2d, lr, binary, conv_width, drop_prob, batch_norm, first_layer_out_channels,
                 first_layer_kernel_size, train_delta, train_crop, test_crop, normalize_x, normalize_y):
    model_name = "{}_{}_lr:{}_conv:{}_drop:{}_traindelta:{}_traincrop:{}_testcrop:{}_nx:{}_ny:{}".format(
              class_name, model_type + ('2d' if is_2d else ''), lr, conv_width, drop_prob, train_delta, train_crop, test_crop, normalize_x, normalize_y)
    if not binary:
        model_name += "_cont"
    if not batch_norm:
        model_name += "_nobatch"
    if first_layer_out_channels is not None:
        model_name += "_first:{},{}".format(first_layer_out_channels, first_layer_kernel_size)
    return model_name

def try_to_load(model, optim, scheduler, output, f, name="checkpoint.pt"):
    try:
        
        checkpoint = torch.load(os.path.join(output, name))
        print(model,checkpoint['state_dict'])
        model.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['opt_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_dict'])
        epoch_resume = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        f.write("Resuming from epoch {}\n".format(epoch_resume))
        print("Resuming from epoch {}\n".format(epoch_resume))
    except FileNotFoundError:
        f.write("Starting run from scratch\n")
        epoch_resume = 0
        best_score = 0

    return epoch_resume, best_score

def run_epoch(split, loader, model, optim, epoch, binary, f, device, cutoff=None, log=True):

    if split == 'train':
        model.train()
    else:
        model.eval()

    losses, ys, yhs = [], [], []

    if log:
        print("epoch {} {}".format(epoch, split))
        print(torch.cuda.max_memory_allocated())
        print(torch.cuda.max_memory_cached())

    for (x, y) in tqdm.tqdm(loader):
        x= x.reshape(-1,5000,12)
        x = np.transpose(x,(0,2,1))
        x = x.to(device).double()

        #x = x.to(device)
        y = y.to(device)
        (y, yh, loss) = model.module.train_step(x, y)
        # print(y.shape,yh.shape)
        if split=='train':
            optim.zero_grad()
            loss.float().backward()
            optim.step()

        losses.append(loss.item())
        ys.extend(y.data.tolist())
        yhs.extend(yh.data.tolist())  

    ys, yhs, losses = np.array(ys), np.array(yhs), np.array(losses)
    try:
        if cutoff is not None:
            score = sklearn.metrics.roc_auc_score(cutoff <= ys, yhs)
        else:
            score = sklearn.metrics.r2_score(ys, yhs)
    except ValueError:
        score = 10000000

    if log:
        f.write("{}, {}, {}, {} \n".format(
            epoch,
            split,
            np.mean(losses),
            score))

        f.flush()

    return losses, ys, yhs, score

def save_model(losses, epoch, model, optim, scheduler, output, score, best_score, save_all):
    loss = np.mean(losses)
    best = False
    if best_score < score:
        best_score = score
        best = True

    # Save checkpoint
    save = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_score': best_score,
        'loss': loss,
        'opt_dict': optim.state_dict(),
        'scheduler_dict': scheduler.state_dict(),
    }

    torch.save(save, os.path.join(output, "checkpoint.pt"))
    if best:
        torch.save(save, os.path.join(output, "best.pt"))
    if save_all:
        torch.save(save, os.path.join(output, "checkpoint{}.pt".format(epoch)))
    return best_score


if __name__ == '__main__':
    class_name = 'hemoglobin'
    lr = 1e-5
    model_name = 'lancet'
    binary=False
    train(class_name=class_name, lr=lr, model_type=model_name, binary=binary)
