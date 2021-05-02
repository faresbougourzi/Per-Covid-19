
import torch
# import torchvision
# from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import cv2
import csv
import os
from skimage import  transform 
import scipy.io as sio
# import dlib
import numpy as np
import cv2
import os

import tqdm as tqdm 
import matplotlib.pyplot as plt
import xlrd

from sklearn.model_selection import train_test_split, KFold




 
database_path = '/data/bougourzi/Covid percentage/Cov-19 percentage-20210327T184035Z-001/Database1/Images'
excel_path = "/data/bougourzi/Covid percentage/Cov-19 percentage-20210327T184035Z-001/Database1/Database_new.xlsx"

input_workbook = xlrd.open_workbook(excel_path)
data_excel = input_workbook.sheet_by_index(0)

images_name = []
covid_per = []
sub_name = []
folds = []

for i in range(data_excel.nrows):
    images_name.append(data_excel.cell_value(i,0))
    covid_per.append(data_excel.cell_value(i,1))
    sub_name.append(data_excel.cell_value(i,3))
    folds.append(data_excel.cell_value(i,2))


Training_data = []
Training_label = []
Fold1 = []
Fold2 = []
Fold3 = []
Fold4 = []
Fold5 = []

i = -1
for line in images_name:
    i += 1
    img_name= line
    full_path_image = os.path.join(database_path, img_name)
    img = cv2.imread(full_path_image)

    Training_data.append(np.array(img))
    Training_label.append(float(covid_per[i]))
    if folds[i] == 1:
        Fold1.append(i)
    elif folds[i] == 2:
        Fold2.append(i)
    elif folds[i] == 3:
        Fold3.append(i) 
    elif folds[i] == 4:
        Fold4.append(i)
    elif folds[i] == 5:
        Fold5.append(i)

 

################## 1 ###############################
train_indx1 = Fold2 + Fold3 + Fold4 + Fold5  
X_train = [Training_data[i] for i in train_indx1]  
y_train = [Training_label[i] for i in train_indx1]  
sub_train = [sub_name[i] for i in train_indx1] 

training= (X_train, y_train, sub_train)
torch.save(training,'train_fold11.pt') 

X_test = [Training_data[i] for i in Fold1]  
y_test = [Training_label[i] for i in Fold1] 
sub_test = [sub_name[i] for i in Fold1] 

training= (X_test, y_test, sub_test)
torch.save(training,'test_fold11.pt')       
            
################### 2 ############################

train_indx2 = Fold1 + Fold3 + Fold4 + Fold5   

X_train = [Training_data[i] for i in train_indx2]  
y_train = [Training_label[i] for i in train_indx2]  
sub_train = [sub_name[i] for i in train_indx2] 

training= (X_train, y_train, sub_train)
torch.save(training,'train_fold21.pt') 
    

X_test = [Training_data[i] for i in Fold2]  
y_test = [Training_label[i] for i in Fold2] 
sub_test = [sub_name[i] for i in Fold2] 

training= (X_test, y_test, sub_test)
torch.save(training,'test_fold21.pt') 

################### 3 ###########################
train_indx3 = Fold1 + Fold2 + Fold4 + Fold5   

X_train = [Training_data[i] for i in train_indx3]  
y_train = [Training_label[i] for i in train_indx3]  
sub_train = [sub_name[i] for i in train_indx3] 

training= (X_train, y_train, sub_train)
torch.save(training,'train_fold31.pt') 
    

X_test = [Training_data[i] for i in Fold3]  
y_test = [Training_label[i] for i in Fold3] 
sub_test = [sub_name[i] for i in Fold3] 

training= (X_test, y_test, sub_test)
torch.save(training,'test_fold31.pt') 

################## 4 ############################
train_indx4 = Fold1 + Fold2 + Fold3 + Fold5   

X_train = [Training_data[i] for i in train_indx4]  
y_train = [Training_label[i] for i in train_indx4]  
sub_train = [sub_name[i] for i in train_indx4] 

training= (X_train, y_train, sub_train)
torch.save(training,'train_fold41.pt') 
    

X_test = [Training_data[i] for i in Fold4]  
y_test = [Training_label[i] for i in Fold4] 
sub_test = [sub_name[i] for i in Fold4] 

training= (X_test, y_test, sub_test)
torch.save(training,'test_fold41.pt') 

#################### 5 ##########################

train_indx5 = Fold1 + Fold2 + Fold3 + Fold4   

X_train = [Training_data[i] for i in train_indx5]  
y_train = [Training_label[i] for i in train_indx5]  
sub_train = [sub_name[i] for i in train_indx5] 

training= (X_train, y_train, sub_train)
torch.save(training,'train_fold51.pt') 
    

X_test = [Training_data[i] for i in Fold5]  
y_test = [Training_label[i] for i in Fold5] 
sub_test = [sub_name[i] for i in Fold5] 

training= (X_test, y_test, sub_test)
torch.save(training,'test_fold51.pt') 

##############################################

