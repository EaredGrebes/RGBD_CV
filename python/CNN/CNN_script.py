import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2 
from random import shuffle 
import seaborn as sns
import gc

# custom functions
import data_loading_functions as datFun
import CNN_functions as cnn

np.random.seed(1)
torch.manual_seed(1)

plt.close('all')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

script_folder_path = os.path.dirname(os.path.abspath(__file__))

#------------------------------------------------------------------------------
# configuration
load_data   = True  # if you run the script in the same worspace, you don't need to load the data each time
force_train = True

data_filename = data_folder = '../../data/training_data.npz'

# for quicker code development, use MNIST
config1 = {
'name':          'My_CNNModel',
'epochs':        100,
'learning_rate': 0.00001,
'optimizer':     'Adam',
'img_size':      15,
'batch_size':    64,
'data_folder':   data_filename,
'model':         cnn.My_CNNModel}

# select which configuration to run
#config = CD_config
config = config1

#------------------------------------------------------------------------------
# load data
val_size = 512
test_size = 512

if load_data:
    print('loading data')
    # train data is batched, val and test are not
    train_data,    \
    train_targets, \
    val_data,      \
    val_targets,   \
    test_data,     \
    test_targets = datFun.load_data(config1['batch_size'], val_size, test_size, data_filename, device)
    
    
#------------------------------------------------------------------------------
# load model (or train from scratch)
 
nnModel, \
loss_val_list, \
loss_train_list = cnn.load_model(train_data, train_targets, val_data, val_targets, config, device, force_train)    

# plot loss and accuracy curves from training
plt.figure()
plt.plot(loss_train_list)
plt.plot(loss_val_list)
plt.title('Loss')
plt.xlabel('training epoch')
plt.legend(['training loss (single batch)', 'validation loss'])

y_test_pred = nnModel(test_data.to(device))