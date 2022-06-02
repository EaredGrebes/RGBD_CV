import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from random import shuffle 
import seaborn as sns
from os.path import exists
from torchvision import models

#------------------------------------------------------------------------------
# Models

# example model given in lab notes
class My_CNNModel(nn.Module):
    def __init__(self, im_d):
        super(My_CNNModel, self).__init__()
        
        self.im_d = im_d # dimension of a feature image
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, 
                              kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        
        # self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, 
        #                       kernel_size=3, stride=1, padding=1)
        # self.relu2 = nn.ReLU()
        
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
     
        # Convolution 2
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, 
                              kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        
        # Fully connected 1
        self.fc1 = nn.Linear(32 * 7 * 7 * 2, 1568) 
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(1568, 2) 
        
    def forward(self, x):
        
        # split image left and right
        xl = x[:,:,:,:self.im_d]
        xr = x[:,:,:,self.im_d:]
        
        # img 1
        xl = self.cnn1(xl)     
        xl = self.relu1(xl) 
        #xl = self.cnn2(xl)     
        #xl = self.relu2(xl) 
        xl = self.maxpool(xl)
        xl = self.cnn3(xl)     
        xl = self.relu3(xl) 
        
        # img 1
        xr = self.cnn1(xr)     
        xr = self.relu1(xr) 
        #xr = self.cnn2(xr)     
        #xr = self.relu2(xr) 
        xr = self.maxpool(xr)
        xr = self.cnn3(xr)     
        xr = self.relu3(xr) 
        
    
        #Flatten
        out = torch.cat((xl.view(xl.size(0), -1), 
                          xr.view(xr.size(0), -1)), axis = 1)
        #out = xl.view(xl.size(0), -1)           

        #Dense
        out = self.fc1(out) 
        out = self.relu4(out)
        out = self.fc2(out)
        
        return out

#------------------------------------------------------------------------------
# helper functions related to models

def load_model(train_data, train_targets, val_data, val_targets, config, device, force_train):
    nnModel = config['model'](config['img_size'])
    model_filename = config['name'] + '.pt'
    
    # train from scratch    
    if force_train or not exists(model_filename):
        print('training model')
        loss_val_list, \
        loss_train_list    = train_model(nnModel, \
                                        train_data, \
                                        train_targets, \
                                        val_data, \
                                        val_targets, \
                                        config, \
                                        device)
            
        torch.save(nnModel.state_dict(), model_filename)
        
        np.savez(config['name'], 
                 loss_val_list     = np.array(loss_val_list), 
                 loss_train_list   = np.array(loss_train_list) )
    
    # check to see if model has already been saved
    else:
        nnModel.load_state_dict(torch.load(model_filename))
        nnModel = nnModel.to(device)
        print('Model loaded.')
        
        filez = np.load(config['name'] + '.npz', allow_pickle=True)
        loss_val_list = filez['loss_val_list']
        loss_train_list = filez['loss_train_list']
        accuracy_val_list = filez['accuracy_val_list']
        
    return nnModel, loss_val_list, loss_train_list     
    

def train_model(nnModel, train_data, train_targets, val_data, val_targets, config, device):

    nnModel = nnModel.to(device)
    val_data = val_data.to(device)
    val_targets = val_targets.to(device)
      
    opt_dict = {'SGD': torch.optim.SGD(nnModel.parameters(),   lr = config['learning_rate']),
                'Adam': torch.optim.Adam(nnModel.parameters(), lr = config['learning_rate'])}
       
    optimizer = opt_dict[config['optimizer']]
    loss_fn   = nn.MSELoss()
    
    # train model
    loss_train_list   = np.zeros((config['epochs'],))
    loss_val_list     = np.zeros((config['epochs'],))
    
    # epoch - one loop through all the data points
    for epoch in tqdm.trange(config['epochs']):
    #for epoch in range(cfg['epochs']):
        
        # some layers (like dropout) have different behavior when training and 
        # evaluating the model, switch to train mode
        nnModel.train()
        
        # batch update the weights
        for X_train, y_train in zip(train_data, train_targets):
            
            #X_train = X_train.to(device)
            #y_train = y_train.to(device)
            
            optimizer.zero_grad()
            y_pred = nnModel(X_train)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()
            
        # evaluate loss and accuracy on validation data
        with torch.no_grad():
            
            nnModel.eval()
            y_val_pred = nnModel(val_data) 
            loss_val = loss_fn(y_val_pred, val_targets)
            
            print(f'loss_val: {loss_val.item()}')
        
        loss_val_list[epoch] = loss_val.detach().item()
        loss_train_list[epoch] = loss.detach().item()
        
    # fee up GPU mem
    del train_data
    del train_targets
    del X_train
    del y_train
    
    return loss_val_list, loss_train_list