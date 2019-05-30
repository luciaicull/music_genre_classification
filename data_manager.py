'''
data_manager.py
A file that loads saved features and convert them into PyTorch DataLoader.
'''

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

random.seed(1234)
np.random.seed(1234)

# Class based on PyTorch Dataset
class GTZANDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        #print(self.x[index].shape, self.y[index].shape)     #(1,1024,128), ()

        #return self.augment_data(self.x[index]), self.y[index]
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]
    '''
    def augment_data(self, data):
        n = np.random.randint(0, 3)
        if n == 0:
            return data
        elif n == 1:
            return self.freq_mask(data, num_mask=2)
        elif n == 2:
            return self.time_mask(data)
        elif n == 3:
            return self.freq_mask(self.time_mask(data))

    def freq_mask(self, data, F=20, num_mask=1):
        #cloned = data.clone()   #(1,1024,128)
        cloned = data.transpose(0,2,1)  #(1,128,1024)
        num_mel_channels = cloned.shape[1]

        for i in range(0, num_mask):
            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)

            if f_zero == f_zero + f:
                return cloned.transpose(0,2,1)    #(1,1024,128)

            mask_end = random.randrange(f_zero, f_zero+f)
            cloned[0][f_zero:mask_end] = 0

        return cloned.transpose(0,2,1)

    def time_mask(self, data, T=20, num_mask=1):
        #cloned = data.clone()   #(1,1024,128)
        #print(type(data))
        cloned = data.transpose(0,2,1)  # (1,128,1024)
        len_spectro = cloned.shape[2]

        for i in range(0, num_mask):
            t = random.randrange(0, T)
            t_zero = random.randrange(0, len_spectro - t)

            if t_zero == t_zero + t:
                return cloned.transpose(0,2,1)        #(1,1024,128)

            mask_end = random.randrange(t_zero, t_zero + t)
            cloned[0][:, t_zero:mask_end] = 0

        return cloned.transpose(0,2,1)
    '''

# Function to get genre index for the given file
def get_label(file_name, hparams):
    genre = file_name.split('.')[0]         # file_name = (genre).(number).wav
    label = hparams.genres.index(genre)

    return label

# Function for loading entire data from given dataset and return numpy array
def load_dataset(set_name, hparams):
    x = []
    y = []

    dataset_path = os.path.join(hparams.feature_path, set_name)

    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            data = np.load(os.path.join(root, file_name))
            # if model is 2D convolution
            #data = np.expand_dims(data, axis=0)

            label = get_label(file_name, hparams)
            x.append(data)  # x = [array(data), array(data), ...]
            y.append(label)  # y = [label1, label2, ...]

    x = np.stack(x)    # x = [[data], [data], ...] (np.ndarray)
    y = np.stack(y)    # y = [label1, label2, ...] (np.ndarray)

    return x, y

# Function to load numpy data and normalize, it returns dataloader for train, valid, test
def get_dataloader(hparams):
    x_train, y_train = load_dataset('train', hparams)
    x_valid, y_valid = load_dataset('valid', hparams)
    x_test, y_test = load_dataset('test', hparams)

    mean = np.mean(x_train)
    std = np.std(x_train)
    x_train = (x_train - mean) / std
    x_valid = (x_valid - mean) / std
    x_test = (x_test - mean) / std

    train_set = GTZANDataset(x_train, y_train)
    valid_set = GTZANDataset(x_valid, y_valid)
    test_set = GTZANDataset(x_test, y_test)

    train_loader = DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=hparams.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=hparams.batch_size, shuffle=False, drop_last=False)

    return train_loader, valid_loader, test_loader