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
# data loading

# main data loading function
def load_data(train_batch_size, val_size, test_size, data_filenames, device):
    
    X_list = []
    y_list = []
    for file in data_filenames:
        filez = np.load(file, allow_pickle=True)
        y_i = filez['y_labels_list']
        X_i = filez['X_feature_data_list']
        X_list.append(X_i)
        y_list.append(y_i)
        
    X_feature_data = X_list[0]
    y_labels       = y_list[0]
    for ii in range(1,len(data_filenames)):
        X_feature_data = np.concatenate((X_feature_data, X_list[ii]), axis = 0)
        y_labels       = np.concatenate((y_labels,       y_list[ii]), axis = 0)
    
    print(y_labels.shape)
    print(X_feature_data.shape)
    N_data, c, h, w = X_feature_data.shape
    train_size = N_data - val_size - test_size
    
    print(f'number of samples: {N_data}')
    idx = np.random.permutation(N_data)
    X_feature_data = X_feature_data[idx,:,:,:].astype(np.float32) / 255
    y_labels = y_labels[idx,:].astype(np.float32)
    
    y_labels = (y_labels - np.mean(y_labels, axis =0)[None,:]) / np.std(y_labels, axis =0)[None,:]
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
    

#------------------------------------------------------------------------------
# visualization

def plot_features(X, y, y_pred, idx, dim):
    
    im1_i = np.stack((X[idx, 0, :, :dim], X[idx, 1, :, :dim], X[idx, 2, :, :dim]), axis = 2)
    im2_i = np.stack((X[idx, 0, :, dim:], X[idx, 1, :, dim:], X[idx, 2, :, dim:]), axis = 2)
    
    y_i = y[idx,:]
    ypred_i = y_pred[idx,:]
    
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(im1_i)
    ax[1].imshow(im2_i)
    
    ax[2].plot(y_i[0], y_i[1], 'go')
    ax[2].plot(ypred_i[0], ypred_i[1], 'ro')
    ax[2].grid('on')
    ax[2].set_xlim([-6, 6])
    ax[2].set_ylim([-6, 6])
    ax[2].set_aspect('equal', 'box')
    ax[2].invert_yaxis()
    ax[2].set_xlabel('pixel dx')
    ax[2].set_ylabel('pixel dy')
    ax[2].legend(['label','CNN prediction'])
    
    fig.tight_layout()
