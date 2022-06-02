import numpy as np
import matplotlib.pyplot as plt
import tqdm
from functools import partial
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2  
import seaborn as sns
from os.path import exists
import time

#------------------------------------------------------------------------------
# Cats and Dogs functions

# main data loading function
def load_data(train_batch_size, val_size, test_size, data_filename, device):
    
    filez = np.load(data_filename, allow_pickle=True)
    y_labels = filez['y_labels_list']
    X_feature_data = filez['X_feature_data_list']
    
    print(y_labels.shape)
    print(X_feature_data.shape)
    N_data, c, h, w = X_feature_data.shape
    train_size = N_data - val_size - test_size
    
    idx = np.random.permutation(N_data)
    X_feature_data = X_feature_data[idx,:,:,:].astype(np.float32) / 255
    y_labels = y_labels[idx,:].astype(np.float32)
    
    # split data into train, validation, and test
    X_train = X_feature_data[0:train_size,:,:,:]
    X_val   = X_feature_data[train_size:train_size + val_size,:,:,:]
    X_test  = X_feature_data[train_size + val_size:train_size + val_size + test_size,:,:,:]
    
    y_train = y_labels[0:train_size,:]
    y_val   = y_labels[train_size:train_size + val_size,:]
    y_test  = y_labels[train_size + val_size:train_size + val_size + test_size,:]

    # torch batches
    train_data, train_targets = torch_batches(X_train, y_train, train_batch_size, device)
    
    # val and test data are single batches
    val_data    = torch.from_numpy(X_val)
    val_targets = torch.from_numpy(y_val)
    test_data    = torch.from_numpy(X_test)
    test_targets = torch.from_numpy(y_test)
    
    return train_data, train_targets, val_data, val_targets, test_data, test_targets


def torch_batches(X_train, y_train, batch_size, device):
    
    N, c, h, w = X_train.shape
    
    if batch_size < N:
        steps = np.floor(N/batch_size).astype(int)
        intervals = np.arange(0, (steps+1)*batch_size+1, batch_size)
        intervals[-1] = N
    else:
        intervals = np.array([0, N])
    
    torch_data = []
    torch_targets = []
    for batch in range(1, len(intervals)):
        
        batch_indexes = np.arange(intervals[batch-1], intervals[batch])
        X_tensor = torch.from_numpy(X_train[batch_indexes,:,:,:]).to(device)
        y_vector = torch.from_numpy(y_train[batch_indexes,:]).to(device)
         
        torch_data.append(X_tensor)
        torch_targets.append(y_vector)
    
    return torch_data, torch_targets
    

    
