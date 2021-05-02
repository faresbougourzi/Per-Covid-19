#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 10:47:37 2021

@author: bougourzi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 00:53:20 2021

@author: bougourzi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# Face Beauty Project
# Bougourzi Fares
from Covid_Per import Covid_Per
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision
# import numpy as np

import torchvision.transforms as transforms
import numpy as np
import scipy.io as sio

from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
  





#############################################################################
#############################################################################
###### Fold1
#############################################################################
#############################################################################
torch.set_grad_enabled(True)
torch.set_printoptions(linewidth=120)    


def smooth_l1_loss(input, target, beta):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2, n + 0.5 * beta**2 - beta)

    return loss.mean()


def huber_loss(input, target, beta):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n <= beta
    loss = torch.where(cond, 0.5 * n ** 2, beta*n - 0.5 * beta**2)

    return loss.mean()


def MAE_distance(preds, labels):
    return torch.sum(torch.abs(preds - labels))

def Adaptive_loss(preds, labels, sigma):
    mse = (1+sigma)*((preds - labels)**2)
    mae = sigma + (torch.abs(preds - labels))
    return torch.mean(mse/mae)

def PC_mine(preds, labels):
    dem = np.sum((preds - np.mean(preds))*(labels - np.mean(labels)))
    mina = (np.sqrt(np.sum((preds - np.mean(preds))**2)))*(np.sqrt(np.sum((labels - np.mean(labels))**2)))
    return dem/mina 


train_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize((244,244)),
        transforms.RandomCrop((244,244)),
        transforms.RandomRotation(degrees = (-10,10)),
        transforms.ToTensor()
])    
#  transforms.      transforms.RandomRotation(degrees = (-10,10)),
test_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize((244,244)),
        transforms.ToTensor()
]) 
train_set = Covid_Per(
        root='./'
        ,train = 'train_fold11.pt'
        ,transform = train_transform
)
#                
test_set = Covid_Per(
        root='./'
        ,train = 'test_fold11.pt'
        ,transform = test_transform
) 

device = torch.device("cuda:2") 

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 20, shuffle = True)
train_loader_ts = torch.utils.data.DataLoader(train_set, batch_size = 1)       
validate_loader = torch.utils.data.DataLoader(test_set, batch_size = 1)

model =  torchvision.models.resnext50_32x4d(pretrained=True)
model.fc = nn.Linear(2048, 1)
# device = torch.device("cuda:0") 
model = model.to(device)     

train_MAE = []
train_RMSE = []
train_PC = []

train_MAE_sub = []
train_RMSE_sub = []
train_PC_sub = []


test_MAE = []    
test_RMSE = []
test_PC = []

test_MAE_sub = []
test_RMSE_sub = []
test_PC_sub = []
epoch_count = []

criterion = nn.MSELoss()
# criterion = huber_loss
sigma = 2

beta_max = 20
beta_min = 3

pc_best = -2
name = './Models/Rex_1_mse_best.pt'
##################################   20 0.0001  ############################################################    
for epoch in range(30):
    epoch_count.append(epoch)
    lr = 0.0001
    if epoch>9:
        lr = 0.00001  
    if epoch>19:
        lr = 0.000001 
    # optimizer = optim.Adam(model.parameters(), lr = lr)
    beta = beta_min + (1/2)* (beta_max - beta_min ) * (1+ np.cos (np.pi * ((epoch+1)/ 30)))

    optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    total_loss = 0
    test_loss = 0
    total_correct = 0
    total_correct_val = 0
    total_distance = 0
    tr_sub = []
    ts_sub = []
    
    for batch in tqdm(train_loader):        
        images, labels, sub = batch
        images = images.float().to(device)
        labels = labels.float().to(device)
        torch.set_grad_enabled(True)
        model.train()
        preds = model(images)
        loss = criterion(preds.squeeze(1), labels)           
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()            
        del images; del labels
        
    itr = -1
    labels2_tr = np.zeros([len(train_set),1])
    labels_pred_tr = np.zeros([len(train_set),1])
    
                     
    for batch in tqdm(train_loader_ts):
        itr += 1            
        images, labels, sub = batch
        images = images.float().to(device)
        labels = labels.float().to(device)
        model.eval()
        with torch.no_grad():
            preds = model(images)
            
        loss = criterion(preds.squeeze(1), labels)               

        total_loss += loss.item()            
        total_correct += MAE_distance(preds.squeeze(1), labels)
        labels2_tr[itr,0] = labels
        labels_pred_tr[itr,0] = preds 
        tr_sub.append(sub)
        del images; del labels            
        
    itr = -1
    labels2_ts = np.zeros([len(test_set),1])
    labels_pred_ts = np.zeros([len(test_set),1])
    for batch in tqdm(validate_loader): 
        itr += 1
        images, labels, sub = batch
        images = images.float().to(device)
        labels = labels.float().to(device)
        model.eval()
        with torch.no_grad():
            preds = model(images)

        loss = criterion(preds.squeeze(1), labels)                

        test_loss += loss.item()            
        total_correct_val += MAE_distance(preds.squeeze(1), labels)
        labels2_ts[itr,0] = labels
        labels_pred_ts[itr,0] = preds
        ts_sub.append(sub)                    
        del images; del labels
        
    test_MAE.append(float(np.mean(np.abs(labels_pred_ts - labels2_ts))))
    test_RMSE.append(float(np.sqrt(np.mean((labels_pred_ts - labels2_ts)**2))))
    test_PC.append(float(PC_mine(labels_pred_ts, labels2_ts)))
    
    train_MAE.append(float(np.mean(np.abs(labels_pred_tr - labels2_tr))))
    train_RMSE.append(float(np.sqrt(np.mean((labels_pred_tr - labels2_tr)**2))))
    train_PC.append(float(PC_mine(labels_pred_tr, labels2_tr)))
    
##################
    tr_sub = np.array(tr_sub)
    tr_subj = list(set(tr_sub))
    sub_perc_label = np.zeros([len(tr_subj),1])
    sub_perc_pred = np.zeros([len(tr_subj),1])
    for i in range(len(tr_subj)):
        tr_indx_i = [ii for ii, e in enumerate(tr_sub) if e == tr_sub[i]]
        llm  = np.mean(labels2_tr[tr_indx_i,0])
        sub_perc_label[i,0] = llm
        lllm = np.mean(labels_pred_tr[tr_indx_i,0])
        sub_perc_pred[i,0] = lllm
    tr_mae_subj =  float(np.mean(np.abs(sub_perc_label - sub_perc_pred)))   
    train_MAE_sub.append(tr_mae_subj) 
    train_RMSE_sub.append(float(np.sqrt(np.mean((sub_perc_label - sub_perc_pred)**2))))
    train_PC_sub.append(float(PC_mine(sub_perc_label , sub_perc_pred)))
    
#################
    ts_sub = np.array(ts_sub)
    ts_subj = list(set(ts_sub))
    sub_perc_label_ts = np.zeros([len(ts_subj),1])
    sub_perc_pred_ts = np.zeros([len(ts_subj),1])
    for i in range(len(ts_subj)):
        ts_indx_i = [ii for ii, e in enumerate(ts_sub) if e == ts_subj[i]]
        llm = np.mean(labels2_ts[ts_indx_i,0])
        sub_perc_label_ts[i,0] = llm
        lllm = np.mean(labels_pred_ts[ts_indx_i,0])
        sub_perc_pred_ts[i,0] = lllm
    ts_mae_subj = float(np.mean(np.abs(sub_perc_label_ts - sub_perc_pred_ts)))
    test_MAE_sub.append(ts_mae_subj) 
    test_RMSE_sub.append(float(np.sqrt(np.mean((sub_perc_label_ts- sub_perc_pred_ts)**2))))
    test_PC_sub.append(float(PC_mine(sub_perc_label_ts, sub_perc_pred_ts)))
    
    
    
    print('Ep: ', epoch, 'PC_tr: ', PC_mine(labels_pred_tr, labels2_tr), 'PC_ts: ',  PC_mine(labels_pred_ts, labels2_ts),'MAE_tr: ', total_correct/len(train_set), 'MAE_ts: ', total_correct_val/len(test_set), 'loss_tr:', total_loss/len(train_set),'loss_ts:', test_loss/len(train_set), 'tr_MAE_s', tr_mae_subj, 'ts_MAE_s', ts_mae_subj)
    pc_best2 = float(PC_mine(labels_pred_ts, labels2_ts))
    if pc_best2 > pc_best: 
        pc_best = pc_best2
        mae_best = float(np.mean(np.abs(labels_pred_ts - labels2_ts)))
        rmse_best = float(np.sqrt(np.mean((labels_pred_ts - labels2_ts)**2)))
        torch.save(model.state_dict(), name)
                        
print(pc_best) 
print(mae_best) 
print(rmse_best) 

#############################################################################
name = './Models/Rex_1_mse.pt'          
torch.save(model.state_dict(), name)

print('Image')
print(float(np.mean(np.abs(labels_pred_ts - labels2_ts))))
print(float(np.sqrt(np.mean((labels_pred_ts - labels2_ts)**2))))
print(float(PC_mine(labels_pred_ts, labels2_ts)))
print('Subj')
print(float(np.mean(np.abs(sub_perc_label_ts - sub_perc_pred_ts))))
print(float(np.sqrt(np.mean((sub_perc_label_ts- sub_perc_pred_ts)**2))))
print(float(PC_mine(sub_perc_label_ts, sub_perc_pred_ts)))

l = (epoch_count, train_MAE,train_RMSE,train_PC,train_MAE_sub, train_RMSE_sub, train_PC_sub,test_MAE_sub,test_RMSE_sub,test_PC_sub, test_MAE,test_RMSE,test_PC)
torch.save(l, './Models/Results_Rex_1_mse.pt')

#################################################################
#############################################################################
#############################################################################
###### Fold2
#############################################################################
#############################################################################
train_set = Covid_Per(
        root='./'
        ,train = 'train_fold21.pt'
        ,transform = train_transform
)
#                
test_set = Covid_Per(
        root='./'
        ,train = 'test_fold21.pt'
        ,transform = test_transform
) 

# device = torch.device("cuda:3") 

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 20, shuffle = True)
train_loader_ts = torch.utils.data.DataLoader(train_set, batch_size = 1)       
validate_loader = torch.utils.data.DataLoader(test_set, batch_size = 1)

model =  torchvision.models.resnext50_32x4d(pretrained=True)
model.fc = nn.Linear(2048, 1)
# device = torch.device("cuda:0") 
model = model.to(device)     

train_MAE = []
train_RMSE = []
train_PC = []

train_MAE_sub = []
train_RMSE_sub = []
train_PC_sub = []


test_MAE = []    
test_RMSE = []
test_PC = []

test_MAE_sub = []
test_RMSE_sub = []
test_PC_sub = []
epoch_count = []

criterion = nn.MSELoss()
# criterion = huber_loss
sigma = 2

beta_max = 20
beta_min = 3

pc_best = -2
name = './Models/Rex_2_mse_best.pt'
##################################   20 0.0001  ############################################################    
for epoch in range(30):
    epoch_count.append(epoch)
    lr = 0.0001
    if epoch>9:
        lr = 0.00001  
    if epoch>19:
        lr = 0.000001 
    # optimizer = optim.Adam(model.parameters(), lr = lr)
    beta = beta_min + (1/2)* (beta_max - beta_min ) * (1+ np.cos (np.pi * ((epoch+1)/ 30)))

    optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    total_loss = 0
    test_loss = 0
    total_correct = 0
    total_correct_val = 0
    total_distance = 0
    tr_sub = []
    ts_sub = []
    
    for batch in tqdm(train_loader):        
        images, labels, sub = batch
        images = images.float().to(device)
        labels = labels.float().to(device)
        torch.set_grad_enabled(True)
        model.train()
        preds = model(images)
        loss = criterion(preds.squeeze(1), labels)           
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()            
        del images; del labels
        
    itr = -1
    labels2_tr = np.zeros([len(train_set),1])
    labels_pred_tr = np.zeros([len(train_set),1])
    
                     
    for batch in tqdm(train_loader_ts):
        itr += 1            
        images, labels, sub = batch
        images = images.float().to(device)
        labels = labels.float().to(device)
        model.eval()
        with torch.no_grad():
            preds = model(images)
            
        loss = criterion(preds.squeeze(1), labels)               

        total_loss += loss.item()            
        total_correct += MAE_distance(preds.squeeze(1), labels)
        labels2_tr[itr,0] = labels
        labels_pred_tr[itr,0] = preds 
        tr_sub.append(sub)
        del images; del labels            
        
    itr = -1
    labels2_ts = np.zeros([len(test_set),1])
    labels_pred_ts = np.zeros([len(test_set),1])
    for batch in tqdm(validate_loader): 
        itr += 1
        images, labels, sub = batch
        images = images.float().to(device)
        labels = labels.float().to(device)
        model.eval()
        with torch.no_grad():
            preds = model(images)

        loss = criterion(preds.squeeze(1), labels)                

        test_loss += loss.item()            
        total_correct_val += MAE_distance(preds.squeeze(1), labels)
        labels2_ts[itr,0] = labels
        labels_pred_ts[itr,0] = preds
        ts_sub.append(sub)                    
        del images; del labels
        
    test_MAE.append(float(np.mean(np.abs(labels_pred_ts - labels2_ts))))
    test_RMSE.append(float(np.sqrt(np.mean((labels_pred_ts - labels2_ts)**2))))
    test_PC.append(float(PC_mine(labels_pred_ts, labels2_ts)))
    
    train_MAE.append(float(np.mean(np.abs(labels_pred_tr - labels2_tr))))
    train_RMSE.append(float(np.sqrt(np.mean((labels_pred_tr - labels2_tr)**2))))
    train_PC.append(float(PC_mine(labels_pred_tr, labels2_tr)))
    
##################
    tr_sub = np.array(tr_sub)
    tr_subj = list(set(tr_sub))
    sub_perc_label = np.zeros([len(tr_subj),1])
    sub_perc_pred = np.zeros([len(tr_subj),1])
    for i in range(len(tr_subj)):
        tr_indx_i = [ii for ii, e in enumerate(tr_sub) if e == tr_sub[i]]
        llm  = np.mean(labels2_tr[tr_indx_i,0])
        sub_perc_label[i,0] = llm
        lllm = np.mean(labels_pred_tr[tr_indx_i,0])
        sub_perc_pred[i,0] = lllm
    tr_mae_subj =  float(np.mean(np.abs(sub_perc_label - sub_perc_pred)))   
    train_MAE_sub.append(tr_mae_subj) 
    train_RMSE_sub.append(float(np.sqrt(np.mean((sub_perc_label - sub_perc_pred)**2))))
    train_PC_sub.append(float(PC_mine(sub_perc_label , sub_perc_pred)))
    
#################
    ts_sub = np.array(ts_sub)
    ts_subj = list(set(ts_sub))
    sub_perc_label_ts = np.zeros([len(ts_subj),1])
    sub_perc_pred_ts = np.zeros([len(ts_subj),1])
    for i in range(len(ts_subj)):
        ts_indx_i = [ii for ii, e in enumerate(ts_sub) if e == ts_subj[i]]
        llm = np.mean(labels2_ts[ts_indx_i,0])
        sub_perc_label_ts[i,0] = llm
        lllm = np.mean(labels_pred_ts[ts_indx_i,0])
        sub_perc_pred_ts[i,0] = lllm
    ts_mae_subj = float(np.mean(np.abs(sub_perc_label_ts - sub_perc_pred_ts)))
    test_MAE_sub.append(ts_mae_subj) 
    test_RMSE_sub.append(float(np.sqrt(np.mean((sub_perc_label_ts- sub_perc_pred_ts)**2))))
    test_PC_sub.append(float(PC_mine(sub_perc_label_ts, sub_perc_pred_ts)))
    
    
    
    print('Ep: ', epoch, 'PC_tr: ', PC_mine(labels_pred_tr, labels2_tr), 'PC_ts: ',  PC_mine(labels_pred_ts, labels2_ts),'MAE_tr: ', total_correct/len(train_set), 'MAE_ts: ', total_correct_val/len(test_set), 'loss_tr:', total_loss/len(train_set),'loss_ts:', test_loss/len(train_set), 'tr_MAE_s', tr_mae_subj, 'ts_MAE_s', ts_mae_subj)
    pc_best2 = float(PC_mine(labels_pred_ts, labels2_ts))
    if pc_best2 > pc_best: 
        pc_best = pc_best2
        mae_best = float(np.mean(np.abs(labels_pred_ts - labels2_ts)))
        rmse_best = float(np.sqrt(np.mean((labels_pred_ts - labels2_ts)**2)))
        torch.save(model.state_dict(), name)
                        
print(pc_best) 
print(mae_best) 
print(rmse_best) 

#############################################################################
name = './Models/Rex_2_mse.pt'          
torch.save(model.state_dict(), name)

print('Image')
print(float(np.mean(np.abs(labels_pred_ts - labels2_ts))))
print(float(np.sqrt(np.mean((labels_pred_ts - labels2_ts)**2))))
print(float(PC_mine(labels_pred_ts, labels2_ts)))
print('Subj')
print(float(np.mean(np.abs(sub_perc_label_ts - sub_perc_pred_ts))))
print(float(np.sqrt(np.mean((sub_perc_label_ts- sub_perc_pred_ts)**2))))
print(float(PC_mine(sub_perc_label_ts, sub_perc_pred_ts)))

l = (epoch_count, train_MAE,train_RMSE,train_PC,train_MAE_sub, train_RMSE_sub, train_PC_sub,test_MAE_sub,test_RMSE_sub,test_PC_sub, test_MAE,test_RMSE,test_PC)
torch.save(l, './Models/Results_Rex_2_mse.pt')

#################################################################
#############################################################################
#############################################################################
###### Fold3
#############################################################################
#############################################################################
train_set = Covid_Per(
        root='./'
        ,train = 'train_fold31.pt'
        ,transform = train_transform
)
#                
test_set = Covid_Per(
        root='./'
        ,train = 'test_fold31.pt'
        ,transform = test_transform
) 

# device = torch.device("cuda:0") 

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 20, shuffle = True)
train_loader_ts = torch.utils.data.DataLoader(train_set, batch_size = 1)       
validate_loader = torch.utils.data.DataLoader(test_set, batch_size = 1)

model =  torchvision.models.resnext50_32x4d(pretrained=True)
model.fc = nn.Linear(2048, 1)
# device = torch.device("cuda:0") 
model = model.to(device)     

train_MAE = []
train_RMSE = []
train_PC = []

train_MAE_sub = []
train_RMSE_sub = []
train_PC_sub = []


test_MAE = []    
test_RMSE = []
test_PC = []

test_MAE_sub = []
test_RMSE_sub = []
test_PC_sub = []
epoch_count = []

criterion = nn.MSELoss()
# criterion = huber_loss
sigma = 2

beta_max = 20
beta_min = 3

pc_best = -2
name = './Models/Rex_3_mse_best.pt'
##################################   20 0.0001  ############################################################    
for epoch in range(30):
    epoch_count.append(epoch)
    lr = 0.0001
    if epoch>9:
        lr = 0.00001  
    if epoch>19:
        lr = 0.000001 
    # optimizer = optim.Adam(model.parameters(), lr = lr)
    beta = beta_min + (1/2)* (beta_max - beta_min ) * (1+ np.cos (np.pi * ((epoch+1)/ 30)))

    optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    total_loss = 0
    test_loss = 0
    total_correct = 0
    total_correct_val = 0
    total_distance = 0
    tr_sub = []
    ts_sub = []
    
    for batch in tqdm(train_loader):        
        images, labels, sub = batch
        images = images.float().to(device)
        labels = labels.float().to(device)
        torch.set_grad_enabled(True)
        model.train()
        preds = model(images)
        loss = criterion(preds.squeeze(1), labels)           
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()            
        del images; del labels
        
    itr = -1
    labels2_tr = np.zeros([len(train_set),1])
    labels_pred_tr = np.zeros([len(train_set),1])
    
                     
    for batch in tqdm(train_loader_ts):
        itr += 1            
        images, labels, sub = batch
        images = images.float().to(device)
        labels = labels.float().to(device)
        model.eval()
        with torch.no_grad():
            preds = model(images)
            
        loss = criterion(preds.squeeze(1), labels)               

        total_loss += loss.item()            
        total_correct += MAE_distance(preds.squeeze(1), labels)
        labels2_tr[itr,0] = labels
        labels_pred_tr[itr,0] = preds 
        tr_sub.append(sub)
        del images; del labels            
        
    itr = -1
    labels2_ts = np.zeros([len(test_set),1])
    labels_pred_ts = np.zeros([len(test_set),1])
    for batch in tqdm(validate_loader): 
        itr += 1
        images, labels, sub = batch
        images = images.float().to(device)
        labels = labels.float().to(device)
        model.eval()
        with torch.no_grad():
            preds = model(images)

        loss = criterion(preds.squeeze(1), labels)                

        test_loss += loss.item()            
        total_correct_val += MAE_distance(preds.squeeze(1), labels)
        labels2_ts[itr,0] = labels
        labels_pred_ts[itr,0] = preds
        ts_sub.append(sub)                    
        del images; del labels
        
    test_MAE.append(float(np.mean(np.abs(labels_pred_ts - labels2_ts))))
    test_RMSE.append(float(np.sqrt(np.mean((labels_pred_ts - labels2_ts)**2))))
    test_PC.append(float(PC_mine(labels_pred_ts, labels2_ts)))
    
    train_MAE.append(float(np.mean(np.abs(labels_pred_tr - labels2_tr))))
    train_RMSE.append(float(np.sqrt(np.mean((labels_pred_tr - labels2_tr)**2))))
    train_PC.append(float(PC_mine(labels_pred_tr, labels2_tr)))
    
##################
    tr_sub = np.array(tr_sub)
    tr_subj = list(set(tr_sub))
    sub_perc_label = np.zeros([len(tr_subj),1])
    sub_perc_pred = np.zeros([len(tr_subj),1])
    for i in range(len(tr_subj)):
        tr_indx_i = [ii for ii, e in enumerate(tr_sub) if e == tr_sub[i]]
        llm  = np.mean(labels2_tr[tr_indx_i,0])
        sub_perc_label[i,0] = llm
        lllm = np.mean(labels_pred_tr[tr_indx_i,0])
        sub_perc_pred[i,0] = lllm
    tr_mae_subj =  float(np.mean(np.abs(sub_perc_label - sub_perc_pred)))   
    train_MAE_sub.append(tr_mae_subj) 
    train_RMSE_sub.append(float(np.sqrt(np.mean((sub_perc_label - sub_perc_pred)**2))))
    train_PC_sub.append(float(PC_mine(sub_perc_label , sub_perc_pred)))
    
#################
    ts_sub = np.array(ts_sub)
    ts_subj = list(set(ts_sub))
    sub_perc_label_ts = np.zeros([len(ts_subj),1])
    sub_perc_pred_ts = np.zeros([len(ts_subj),1])
    for i in range(len(ts_subj)):
        ts_indx_i = [ii for ii, e in enumerate(ts_sub) if e == ts_subj[i]]
        llm = np.mean(labels2_ts[ts_indx_i,0])
        sub_perc_label_ts[i,0] = llm
        lllm = np.mean(labels_pred_ts[ts_indx_i,0])
        sub_perc_pred_ts[i,0] = lllm
    ts_mae_subj = float(np.mean(np.abs(sub_perc_label_ts - sub_perc_pred_ts)))
    test_MAE_sub.append(ts_mae_subj) 
    test_RMSE_sub.append(float(np.sqrt(np.mean((sub_perc_label_ts- sub_perc_pred_ts)**2))))
    test_PC_sub.append(float(PC_mine(sub_perc_label_ts, sub_perc_pred_ts)))
    
    
    
    print('Ep: ', epoch, 'PC_tr: ', PC_mine(labels_pred_tr, labels2_tr), 'PC_ts: ',  PC_mine(labels_pred_ts, labels2_ts),'MAE_tr: ', total_correct/len(train_set), 'MAE_ts: ', total_correct_val/len(test_set), 'loss_tr:', total_loss/len(train_set),'loss_ts:', test_loss/len(train_set), 'tr_MAE_s', tr_mae_subj, 'ts_MAE_s', ts_mae_subj)
    pc_best2 = float(PC_mine(labels_pred_ts, labels2_ts))
    if pc_best2 > pc_best: 
        pc_best = pc_best2
        mae_best = float(np.mean(np.abs(labels_pred_ts - labels2_ts)))
        rmse_best = float(np.sqrt(np.mean((labels_pred_ts - labels2_ts)**2)))
        torch.save(model.state_dict(), name)
                        
print(pc_best) 
print(mae_best) 
print(rmse_best) 

#############################################################################
name = './Models/Rex_3_mse.pt'          
torch.save(model.state_dict(), name)

print('Image')
print(float(np.mean(np.abs(labels_pred_ts - labels2_ts))))
print(float(np.sqrt(np.mean((labels_pred_ts - labels2_ts)**2))))
print(float(PC_mine(labels_pred_ts, labels2_ts)))
print('Subj')
print(float(np.mean(np.abs(sub_perc_label_ts - sub_perc_pred_ts))))
print(float(np.sqrt(np.mean((sub_perc_label_ts- sub_perc_pred_ts)**2))))
print(float(PC_mine(sub_perc_label_ts, sub_perc_pred_ts)))

l = (epoch_count, train_MAE,train_RMSE,train_PC,train_MAE_sub, train_RMSE_sub, train_PC_sub,test_MAE_sub,test_RMSE_sub,test_PC_sub, test_MAE,test_RMSE,test_PC)
torch.save(l, './Models/Results_Rex_3_mse.pt')
#################################################################
#############################################################################
#############################################################################
###### Fold4
#############################################################################
#############################################################################
train_set = Covid_Per(
        root='./'
        ,train = 'train_fold41.pt'
        ,transform = train_transform
)
#                
test_set = Covid_Per(
        root='./'
        ,train = 'test_fold41.pt'
        ,transform = test_transform
) 

# device = torch.device("cuda:3") 

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 20, shuffle = True)
train_loader_ts = torch.utils.data.DataLoader(train_set, batch_size = 1)       
validate_loader = torch.utils.data.DataLoader(test_set, batch_size = 1)

model =  torchvision.models.resnext50_32x4d(pretrained=True)
model.fc = nn.Linear(2048, 1)
# device = torch.device("cuda:0") 
model = model.to(device)     

train_MAE = []
train_RMSE = []
train_PC = []

train_MAE_sub = []
train_RMSE_sub = []
train_PC_sub = []


test_MAE = []    
test_RMSE = []
test_PC = []

test_MAE_sub = []
test_RMSE_sub = []
test_PC_sub = []
epoch_count = []

criterion = nn.MSELoss()
# criterion = huber_loss
sigma = 2

beta_max = 20
beta_min = 3

pc_best = -2
name = './Models/Rex_4_mse_best.pt'
##################################   20 0.0001  ############################################################    
for epoch in range(30):
    epoch_count.append(epoch)
    lr = 0.0001
    if epoch>9:
        lr = 0.00001  
    if epoch>19:
        lr = 0.000001 
    # optimizer = optim.Adam(model.parameters(), lr = lr)
    beta = beta_min + (1/2)* (beta_max - beta_min ) * (1+ np.cos (np.pi * ((epoch+1)/ 30)))

    optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    total_loss = 0
    test_loss = 0
    total_correct = 0
    total_correct_val = 0
    total_distance = 0
    tr_sub = []
    ts_sub = []
    
    for batch in tqdm(train_loader):        
        images, labels, sub = batch
        images = images.float().to(device)
        labels = labels.float().to(device)
        torch.set_grad_enabled(True)
        model.train()
        preds = model(images)
        loss = criterion(preds.squeeze(1), labels)           
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()            
        del images; del labels
        
    itr = -1
    labels2_tr = np.zeros([len(train_set),1])
    labels_pred_tr = np.zeros([len(train_set),1])
    
                     
    for batch in tqdm(train_loader_ts):
        itr += 1            
        images, labels, sub = batch
        images = images.float().to(device)
        labels = labels.float().to(device)
        model.eval()
        with torch.no_grad():
            preds = model(images)
            
        loss = criterion(preds.squeeze(1), labels)               

        total_loss += loss.item()            
        total_correct += MAE_distance(preds.squeeze(1), labels)
        labels2_tr[itr,0] = labels
        labels_pred_tr[itr,0] = preds 
        tr_sub.append(sub)
        del images; del labels            
        
    itr = -1
    labels2_ts = np.zeros([len(test_set),1])
    labels_pred_ts = np.zeros([len(test_set),1])
    for batch in tqdm(validate_loader): 
        itr += 1
        images, labels, sub = batch
        images = images.float().to(device)
        labels = labels.float().to(device)
        model.eval()
        with torch.no_grad():
            preds = model(images)

        loss = criterion(preds.squeeze(1), labels)                

        test_loss += loss.item()            
        total_correct_val += MAE_distance(preds.squeeze(1), labels)
        labels2_ts[itr,0] = labels
        labels_pred_ts[itr,0] = preds
        ts_sub.append(sub)                    
        del images; del labels
        
    test_MAE.append(float(np.mean(np.abs(labels_pred_ts - labels2_ts))))
    test_RMSE.append(float(np.sqrt(np.mean((labels_pred_ts - labels2_ts)**2))))
    test_PC.append(float(PC_mine(labels_pred_ts, labels2_ts)))
    
    train_MAE.append(float(np.mean(np.abs(labels_pred_tr - labels2_tr))))
    train_RMSE.append(float(np.sqrt(np.mean((labels_pred_tr - labels2_tr)**2))))
    train_PC.append(float(PC_mine(labels_pred_tr, labels2_tr)))
    
##################
    tr_sub = np.array(tr_sub)
    tr_subj = list(set(tr_sub))
    sub_perc_label = np.zeros([len(tr_subj),1])
    sub_perc_pred = np.zeros([len(tr_subj),1])
    for i in range(len(tr_subj)):
        tr_indx_i = [ii for ii, e in enumerate(tr_sub) if e == tr_sub[i]]
        llm  = np.mean(labels2_tr[tr_indx_i,0])
        sub_perc_label[i,0] = llm
        lllm = np.mean(labels_pred_tr[tr_indx_i,0])
        sub_perc_pred[i,0] = lllm
    tr_mae_subj =  float(np.mean(np.abs(sub_perc_label - sub_perc_pred)))   
    train_MAE_sub.append(tr_mae_subj) 
    train_RMSE_sub.append(float(np.sqrt(np.mean((sub_perc_label - sub_perc_pred)**2))))
    train_PC_sub.append(float(PC_mine(sub_perc_label , sub_perc_pred)))
    
#################
    ts_sub = np.array(ts_sub)
    ts_subj = list(set(ts_sub))
    sub_perc_label_ts = np.zeros([len(ts_subj),1])
    sub_perc_pred_ts = np.zeros([len(ts_subj),1])
    for i in range(len(ts_subj)):
        ts_indx_i = [ii for ii, e in enumerate(ts_sub) if e == ts_subj[i]]
        llm = np.mean(labels2_ts[ts_indx_i,0])
        sub_perc_label_ts[i,0] = llm
        lllm = np.mean(labels_pred_ts[ts_indx_i,0])
        sub_perc_pred_ts[i,0] = lllm
    ts_mae_subj = float(np.mean(np.abs(sub_perc_label_ts - sub_perc_pred_ts)))
    test_MAE_sub.append(ts_mae_subj) 
    test_RMSE_sub.append(float(np.sqrt(np.mean((sub_perc_label_ts- sub_perc_pred_ts)**2))))
    test_PC_sub.append(float(PC_mine(sub_perc_label_ts, sub_perc_pred_ts)))
    
    
    
    print('Ep: ', epoch, 'PC_tr: ', PC_mine(labels_pred_tr, labels2_tr), 'PC_ts: ',  PC_mine(labels_pred_ts, labels2_ts),'MAE_tr: ', total_correct/len(train_set), 'MAE_ts: ', total_correct_val/len(test_set), 'loss_tr:', total_loss/len(train_set),'loss_ts:', test_loss/len(train_set), 'tr_MAE_s', tr_mae_subj, 'ts_MAE_s', ts_mae_subj)
    pc_best2 = float(PC_mine(labels_pred_ts, labels2_ts))
    if pc_best2 > pc_best: 
        pc_best = pc_best2
        mae_best = float(np.mean(np.abs(labels_pred_ts - labels2_ts)))
        rmse_best = float(np.sqrt(np.mean((labels_pred_ts - labels2_ts)**2)))
        torch.save(model.state_dict(), name)
                        
print(pc_best) 
print(mae_best) 
print(rmse_best) 

#############################################################################
name = './Models/Rex_4_mse.pt'          
torch.save(model.state_dict(), name)

print('Image')
print(float(np.mean(np.abs(labels_pred_ts - labels2_ts))))
print(float(np.sqrt(np.mean((labels_pred_ts - labels2_ts)**2))))
print(float(PC_mine(labels_pred_ts, labels2_ts)))
print('Subj')
print(float(np.mean(np.abs(sub_perc_label_ts - sub_perc_pred_ts))))
print(float(np.sqrt(np.mean((sub_perc_label_ts- sub_perc_pred_ts)**2))))
print(float(PC_mine(sub_perc_label_ts, sub_perc_pred_ts)))

l = (epoch_count, train_MAE,train_RMSE,train_PC,train_MAE_sub, train_RMSE_sub, train_PC_sub,test_MAE_sub,test_RMSE_sub,test_PC_sub, test_MAE,test_RMSE,test_PC)
torch.save(l, './Models/Results_Rex_4_mse.pt')
#################################################################
#############################################################################
#############################################################################
###### Fold5
#############################################################################
#############################################################################
train_set = Covid_Per(
        root='./'
        ,train = 'train_fold51.pt'
        ,transform = train_transform
)
#                
test_set = Covid_Per(
        root='./'
        ,train = 'test_fold51.pt'
        ,transform = test_transform
) 

# device = torch.device("cuda:3") 

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 20, shuffle = True)
train_loader_ts = torch.utils.data.DataLoader(train_set, batch_size = 1)       
validate_loader = torch.utils.data.DataLoader(test_set, batch_size = 1)

model =  torchvision.models.resnext50_32x4d(pretrained=True)
model.fc = nn.Linear(2048, 1)
# device = torch.device("cuda:0") 
model = model.to(device)     

train_MAE = []
train_RMSE = []
train_PC = []

train_MAE_sub = []
train_RMSE_sub = []
train_PC_sub = []


test_MAE = []    
test_RMSE = []
test_PC = []

test_MAE_sub = []
test_RMSE_sub = []
test_PC_sub = []
epoch_count = []

criterion = nn.MSELoss()
# criterion = huber_loss
sigma = 2

beta_max = 20
beta_min = 3

pc_best = -2
name = './Models/Rex_5_mse_best.pt'
##################################   20 0.0001  ############################################################    
for epoch in range(30):
    epoch_count.append(epoch)
    lr = 0.0001
    if epoch>9:
        lr = 0.00001  
    if epoch>19:
        lr = 0.000001 
    # optimizer = optim.Adam(model.parameters(), lr = lr)
    beta = beta_min + (1/2)* (beta_max - beta_min ) * (1+ np.cos (np.pi * ((epoch+1)/ 30)))

    optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    total_loss = 0
    test_loss = 0
    total_correct = 0
    total_correct_val = 0
    total_distance = 0
    tr_sub = []
    ts_sub = []
    
    for batch in tqdm(train_loader):        
        images, labels, sub = batch
        images = images.float().to(device)
        labels = labels.float().to(device)
        torch.set_grad_enabled(True)
        model.train()
        preds = model(images)
        loss = criterion(preds.squeeze(1), labels)           
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()            
        del images; del labels
        
    itr = -1
    labels2_tr = np.zeros([len(train_set),1])
    labels_pred_tr = np.zeros([len(train_set),1])
    
                     
    for batch in tqdm(train_loader_ts):
        itr += 1            
        images, labels, sub = batch
        images = images.float().to(device)
        labels = labels.float().to(device)
        model.eval()
        with torch.no_grad():
            preds = model(images)
            
        loss = criterion(preds.squeeze(1), labels)               

        total_loss += loss.item()            
        total_correct += MAE_distance(preds.squeeze(1), labels)
        labels2_tr[itr,0] = labels
        labels_pred_tr[itr,0] = preds 
        tr_sub.append(sub)
        del images; del labels            
        
    itr = -1
    labels2_ts = np.zeros([len(test_set),1])
    labels_pred_ts = np.zeros([len(test_set),1])
    for batch in tqdm(validate_loader): 
        itr += 1
        images, labels, sub = batch
        images = images.float().to(device)
        labels = labels.float().to(device)
        model.eval()
        with torch.no_grad():
            preds = model(images)

        loss = criterion(preds.squeeze(1), labels)                

        test_loss += loss.item()            
        total_correct_val += MAE_distance(preds.squeeze(1), labels)
        labels2_ts[itr,0] = labels
        labels_pred_ts[itr,0] = preds
        ts_sub.append(sub)                    
        del images; del labels
        
    test_MAE.append(float(np.mean(np.abs(labels_pred_ts - labels2_ts))))
    test_RMSE.append(float(np.sqrt(np.mean((labels_pred_ts - labels2_ts)**2))))
    test_PC.append(float(PC_mine(labels_pred_ts, labels2_ts)))
    
    train_MAE.append(float(np.mean(np.abs(labels_pred_tr - labels2_tr))))
    train_RMSE.append(float(np.sqrt(np.mean((labels_pred_tr - labels2_tr)**2))))
    train_PC.append(float(PC_mine(labels_pred_tr, labels2_tr)))
    
##################
    tr_sub = np.array(tr_sub)
    tr_subj = list(set(tr_sub))
    sub_perc_label = np.zeros([len(tr_subj),1])
    sub_perc_pred = np.zeros([len(tr_subj),1])
    for i in range(len(tr_subj)):
        tr_indx_i = [ii for ii, e in enumerate(tr_sub) if e == tr_sub[i]]
        llm  = np.mean(labels2_tr[tr_indx_i,0])
        sub_perc_label[i,0] = llm
        lllm = np.mean(labels_pred_tr[tr_indx_i,0])
        sub_perc_pred[i,0] = lllm
    tr_mae_subj =  float(np.mean(np.abs(sub_perc_label - sub_perc_pred)))   
    train_MAE_sub.append(tr_mae_subj) 
    train_RMSE_sub.append(float(np.sqrt(np.mean((sub_perc_label - sub_perc_pred)**2))))
    train_PC_sub.append(float(PC_mine(sub_perc_label , sub_perc_pred)))
    
#################
    ts_sub = np.array(ts_sub)
    ts_subj = list(set(ts_sub))
    sub_perc_label_ts = np.zeros([len(ts_subj),1])
    sub_perc_pred_ts = np.zeros([len(ts_subj),1])
    for i in range(len(ts_subj)):
        ts_indx_i = [ii for ii, e in enumerate(ts_sub) if e == ts_subj[i]]
        llm = np.mean(labels2_ts[ts_indx_i,0])
        sub_perc_label_ts[i,0] = llm
        lllm = np.mean(labels_pred_ts[ts_indx_i,0])
        sub_perc_pred_ts[i,0] = lllm
    ts_mae_subj = float(np.mean(np.abs(sub_perc_label_ts - sub_perc_pred_ts)))
    test_MAE_sub.append(ts_mae_subj) 
    test_RMSE_sub.append(float(np.sqrt(np.mean((sub_perc_label_ts- sub_perc_pred_ts)**2))))
    test_PC_sub.append(float(PC_mine(sub_perc_label_ts, sub_perc_pred_ts)))
    
    
    
    print('Ep: ', epoch, 'PC_tr: ', PC_mine(labels_pred_tr, labels2_tr), 'PC_ts: ',  PC_mine(labels_pred_ts, labels2_ts),'MAE_tr: ', total_correct/len(train_set), 'MAE_ts: ', total_correct_val/len(test_set), 'loss_tr:', total_loss/len(train_set),'loss_ts:', test_loss/len(train_set), 'tr_MAE_s', tr_mae_subj, 'ts_MAE_s', ts_mae_subj)
    pc_best2 = float(PC_mine(labels_pred_ts, labels2_ts))
    if pc_best2 > pc_best: 
        pc_best = pc_best2
        mae_best = float(np.mean(np.abs(labels_pred_ts - labels2_ts)))
        rmse_best = float(np.sqrt(np.mean((labels_pred_ts - labels2_ts)**2)))
        torch.save(model.state_dict(), name)
                        
print(pc_best) 
print(mae_best) 
print(rmse_best) 

#############################################################################
name = './Models/Rex_5_mse.pt'          
torch.save(model.state_dict(), name)

print('Image')
print(float(np.mean(np.abs(labels_pred_ts - labels2_ts))))
print(float(np.sqrt(np.mean((labels_pred_ts - labels2_ts)**2))))
print(float(PC_mine(labels_pred_ts, labels2_ts)))
print('Subj')
print(float(np.mean(np.abs(sub_perc_label_ts - sub_perc_pred_ts))))
print(float(np.sqrt(np.mean((sub_perc_label_ts- sub_perc_pred_ts)**2))))
print(float(PC_mine(sub_perc_label_ts, sub_perc_pred_ts)))

l = (epoch_count, train_MAE,train_RMSE,train_PC,train_MAE_sub, train_RMSE_sub, train_PC_sub,test_MAE_sub,test_RMSE_sub,test_PC_sub, test_MAE,test_RMSE,test_PC)
torch.save(l, './Models/Results_Rex_5_mse.pt')

