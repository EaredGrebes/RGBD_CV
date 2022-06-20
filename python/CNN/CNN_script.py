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
import time

# custom functions
import helper_functions as datFun
import CNN_functions as cnn

np.random.seed(2)
torch.manual_seed(2)

plt.rcParams.update({'font.size': 10})

# plt.close('all')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

script_folder_path = os.path.dirname(os.path.abspath(__file__))

#------------------------------------------------------------------------------
# configuration
load_data   = True  # if you run the script in the same worspace, you don't need to load the data each time
force_train = True

data_filename = data_folder = '../../data/training_data.npz'
lr = 0.00002
lr = 0.00001
#lr = 0.000015

data_filenames1 = [ \
                  '../../../project_data/02-06-2022-19:27:20/training_data_V2.npz',
                  '../../../project_data/02-06-2022-20:22:50/training_data_V2.npz',
                  '../../../project_data/02-06-2022-20:23:35/training_data_V2.npz',
                  '../../../project_data/02-06-2022-20:24:28/training_data_V2.npz',
                  '../../../project_data/02-06-2022-20:54:16/training_data_V2.npz'] 
    
data_filenames2 = [ \
                  '../../../project_data/03-06-2022-11:26:05/training_data_V2.npz',
                  '../../../project_data/03-06-2022-11:47:24/training_data_V2.npz',
                  '../../../project_data/03-06-2022-12:12:22/training_data_V2.npz',
                  '../../../project_data/03-06-2022-12:13:28/training_data_V2.npz',
                  '../../../project_data/03-06-2022-12:14:11/training_data_V2.npz',
                  '../../../project_data/03-06-2022-12:17:57/training_data_V2.npz',
                  '../../../project_data/03-06-2022-12:20:13/training_data_V2.npz']  
    
    
data_filenames = data_filenames1 + data_filenames2
    
#data_filenames = [ '../../../project_data/02-06-2022-20:54:16/training_data.npz']     

# for quicker code development, use MNIST
config1 = {
'name':          'My_CNNModel',
'epochs':        50,
'learning_rate': lr,
'optimizer':     'Adam',
'img_size':      19,
'batch_size':    128,
'data_folder':   data_filename,
'model':         cnn.My_CNNModel}

config2 = {
'name':          'My_CNNModel_V2',
'epochs':        50,
'learning_rate': lr,
'optimizer':     'Adam',
'img_size':      19,
'batch_size':    128,
'data_folder':   data_filename,
'model':         cnn.My_CNNModel_V2}

config3 = {
'name':          'My_CNNModel_V5',
'epochs':        50,
'learning_rate': lr,
'optimizer':     'Adam',
'img_size':      19,
'batch_size':    128,
'data_folder':   data_filename,
'model':         cnn.My_CNNModel_V5}

config4 = {
'name':          'My_CNNModel_split3',
'epochs':        50,
'learning_rate': lr,
'optimizer':     'Adam',
'img_size':      17,
'batch_size':    256,
'data_folder':   data_filename,
'model':         cnn.My_CNNModel_split}

# select which configuration to run
#config = CD_config
config = config4

#------------------------------------------------------------------------------
# load data
val_size = 1024
test_size = 1024

if load_data:
    print('loading data')
    # train data is batched, val and test are not
    train_data,    \
    train_targets, \
    val_data,      \
    val_targets,   \
    test_data,     \
    test_targets = datFun.load_data(config['batch_size'], val_size, test_size, data_filenames, device)
    
    
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

#------------------------------------------------------------------------------
# evaluate mdoel against test data
nnModel.eval()

# timing evaluation
# with torch.no_grad():
#     timerVec = []
#     for X_train in train_data:
#         X_train.to(device)
#         tic = time.perf_counter()
#         y = nnModel(X_train) 
#         timerVec.append(time.perf_counter() - tic)
#     print(np.array(timerVec).mean())
        

y_test_pred = nnModel(test_data.to(device)).to('cpu').detach().numpy()
y_test = test_targets.numpy()
y_err = y_test_pred - y_test

y_test_mag = np.sqrt(np.sum(y_test*y_test, axis = 1))
y_pred_mag = np.sqrt(np.sum(y_test_pred*y_test_pred, axis = 1))
test_ids = np.arange(0, len(y_pred_mag))
test_ids_sub = test_ids[y_test_mag > 3]
#test_ids_sub = np.arange(0,10)

X_test = test_data.numpy()

for ii in test_ids_sub:
    datFun.plot_features(X_test, y_test, y_test_pred,ii, config['img_size'])

t1 = np.std(y_err, axis =0)
t2 = np.std(y_test, axis = 0)

fig, ax = plt.subplots(1,2)
ax[0].hist(y_test[:,0], 200)
#ax[0].hist(y_err[:,0], 200)
ax[0].set_title('Pixel Offset std: {:.2f} '.format(t2[0]))
#ax[0].legend(['Pixel Offset std: {:.2f} '.format(t2[0]), 'Residual Error after CNN std: {:.2f}'.format(t1[0])])
ax[0].set_xlabel('pixel offset')
ax[0].set_ylabel('count')
ax[0].set_xlim([-5, 5])

#ax[1].hist(y_test[:,0], 200)
ax[1].hist(y_err[:,0], 200)
ax[1].set_title('Residual Error after CNN std: {:.2f}'.format(t1[0]))
#ax[1].legend(['Pixel Offset std: {:.2f} '.format(t2[0]), 'Residual Error after CNN std: {:.2f}'.format(t1[0])])
ax[1].set_xlabel('pixel offset')
ax[1].set_ylabel('count')
ax[1].set_xlim([-5, 5])

fig, ax = plt.subplots(2,1)
ax[0].plot(y_test[:,0])
ax[0].plot(y_test_pred[:,0])
ax[0].set_title('x pixel offset')
ax[0].legend(['label','CNN prediction'])
#ax[0].set_xlabel('sample')
ax[0].set_ylabel('pixels')

ax[1].plot(y_test[:,1])
ax[1].plot(y_test_pred[:,1])
ax[1].set_title('y pixel offset')
ax[1].set_xlabel('sample')
ax[1].set_ylabel('pixels')

