import os

from functools import partial

import numpy as np
import pandas as pd
import torch
import tqdm

signal_means = np.array([[ 4.06242161,  5.33713082,  1.27467398, -4.70654383,  1.39171611,
        4.71275068, -5.97738404, -4.9793293 , -3.7949765 ,  1.97712444,
       -0.63192787, -1.61390998]])

signal_stds = np.array([[ 83.23734076, 107.67419237, 107.04505633,  80.08995803,
        79.56394011,  80.21471818,  96.71616693, 113.48993386,
       102.74763197,  95.96592848,  96.43983475,  93.60954777]])

class ECGDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_csv,
                 data_dir,
                 split='train',
                 signal_length=None,
                 leads=None,
                 in_mem=False,
                 cache=True,
                 labelkey='val',
                 filekey='deid_filename',
                 first_n=None,
                 max_diff=None,
                 binary_cutoff=None,
                 log=False, 
                 y_mean=None,
                 y_std=None,
                 normalize_x="lead"):

        self.signal_length = signal_length
        self.leads = leads
        self.in_mem = in_mem
        self.cache = cache

        self.y_mean = y_mean
        self.y_std = y_std
        self.normalize_x = normalize_x

        self.filelist = pd.read_csv(data_csv)
        self.filelist = self.filelist[self.filelist["split"] == split]
        if max_diff is not None:
            self.filelist = self.filelist[self.filelist['diff'] <= max_diff]

        self.filenames = self.filelist['deid_filename'].apply(
            lambda fname: os.path.join(data_dir, fname + '.npy'), 1)

        if first_n:
            self.filelist = self.filelist[:first_n]
            self.filenames = self.filenames[:first_n]

        n_files = len(self.filenames)
        print("Looking for {} files".format(n_files))
        self.exists = self.filenames.apply(os.path.exists)
        self.missing = self.filenames[~self.exists]
        self.filelist = self.filelist[self.exists]
        self.filenames = np.array(self.filenames[self.exists])
        n_present = len(self.filenames)
        print(data_dir, data_csv)
        print("Found {}".format(n_present))

        self.labels = np.array(self.filelist[labelkey])
        if log:
            self.labels = np.log(1e-5 + self.labels)

        if y_mean is not None:
            self.labels -= y_mean
        if y_std is not None:
            self.labels /= y_std

        assert not np.any(np.isnan(self.labels))

        if binary_cutoff:
            self.labels = self.labels >= binary_cutoff

        if in_mem:
            print("loading all")
            signals = []
            for filename in tqdm.tqdm(self.filenames):
                signals.append(np.load(filename))
            self.signals = signals
        elif cache:
            self.signals = [None] * len(self.filenames)

    def __getitem__(self, index):
        if self.in_mem:
            x = self.signals[index]
        elif self.cache:
            if self.signals[index] is None:
                x = np.load(self.filenames[index])
                self.signals[index] = x
            else:
                x = self.signals[index]
        else:
            x = np.load(self.filenames[index])
        assert len(x) == 5500
        x = x[:5300]

        if self.normalize_x == 'lead':
            x = (x - signal_means) / signal_stds

        if self.leads:
            x = x[:, self.leads]

        if self.signal_length and self.signal_length < 5300:
            start = np.random.randint(5300-self.signal_length)
            x = x[start:start+self.signal_length]

        x = x.T

        y = self.labels[index]

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(np.array(y)).float()

        return x, y


    def __len__(self):
        return len(self.filenames)

def get_stats(dataset, by_lead=True, n=1280, batch_size=128, num_workers=0):

    indices = np.random.choice(len(dataset), n, replace=False)
    dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True) 

    if by_lead:
        x_1 = np.zeros(12)
        x_2 = np.zeros(12)
        ax = 1
        #[128, 12, 5300]
    else:
        x_1 = 0
        x_2 = 0
        ax = None

    y_1 = 0
    y_2 = 0

    n_b = 0

    for x, y in dataloader:
        x = x.numpy().transpose(1, 0, 2).reshape(12, -1) # [128, 12, 5300] -> [12, -1]
        d = x.shape[1] if by_lead else 12 * x.shape[1]
        x_1 += np.sum(x / d, axis=ax)
        x_2 += np.sum(x**2 / d, axis=ax)

        y = y.numpy() 
        y_1 += np.sum(y / np.prod(y.shape))
        y_2 += np.sum(y ** 2 / np.prod(y.shape))

        n_b += 1

    return (x_1 / n_b, (x_2 / n_b - (x_1 / n_b) ** 2) ** .5,
            y_1 / n_b, (y_2 / n_b - (y_1 / n_b) ** 2) ** .5)

if __name__ == '__main__':
    print('initializing')
    d = ECGDataset(
        '/oak/stanford/groups/euan/projects/stanfordECG/labels/A1c_labels.csv',
        '/oak/stanford/groups/euan/projects/stanfordECG/waveforms/',
        'train',
        first_n=1280
    )
    print('computing')
    stats = get_stats(d)
    import pdb; pdb.set_trace()
    print(stats)
