import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
# from sklearn.model_selection import KFold

#Overload pytorch Dataset for Customization
class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, transform=None, target_transfrom=None):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.is_tensor(X):
            self.X = X 
        else:
            self.X = torch.FloatTensor(X).to(device)
        
        if torch.is_tensor(Y):
            self.Y = Y
        else:
            self.Y = torch.LongTensor(Y).to(device)
        self.transform = transform
        self.target_transform = target_transfrom
        
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x,y


#Load Raw Data
def data_loader(path, verbose = False, subjects = [0,1,2,3,4,5,6,7,8]):
    X_test = np.load(path + "X_test.npy")
    y_test = np.load(path + "y_test.npy")
    person_train_valid = np.load(path + "person_train_valid.npy")
    X_train_valid = np.load(path + "X_train_valid.npy")
    y_train_valid = np.load(path + "y_train_valid.npy")
    person_test = np.load(path + "person_test.npy")

    y_train_valid -= 769
    y_test -= 769

    #When we want to load specific subjects
    if len(subjects) != 9:
        X_train_valid_subjects = np.empty(shape=[0, X_train_valid.shape[1], X_train_valid.shape[2]])
        X_test_subjects = np.empty(shape=[0, X_test.shape[1], X_test.shape[2]])
        y_train_valid_subjects = np.empty(shape=[0])
        y_test_subjects = np.empty(shape=[0])

        for person in subjects:
            X_train_valid_tmp = X_train_valid[np.where(person_train_valid == person)[0], :, :]
            X_test_tmp = X_test[np.where(person_test == person)[0], :, :]
            y_train_valid_tmp = y_train_valid[np.where(person_train_valid == person)[0]]
            y_test_tmp = y_test[np.where(person_test == person)[0]]

            X_train_valid_subjects = np.concatenate((X_train_valid_subjects, X_train_valid_tmp), axis=0)
            X_test_subjects = np.concatenate((X_test_subjects, X_test_tmp), axis=0)
            y_train_valid_subjects = np.concatenate((y_train_valid_subjects, y_train_valid_tmp))
            y_test_subjects = np.concatenate((y_test_subjects, y_test_tmp))
        
        if verbose:
            print('The X train & valid data size is {}'.format(X_train_valid_subjects.shape))
            print('The X test data size is {}'.format(X_test_subjects.shape))
        
        return X_train_valid_subjects, X_test_subjects, y_train_valid_subjects, y_test_subjects
    
    else:

        if verbose:
            print('The X train & valid data size is {}'.format(X_train_valid.shape))
            print('The X test data size is {}'.format(X_test.shape))

    return X_train_valid,  y_train_valid, X_test, y_test

#Perform Data Preprocessing
def data_process(X_train, X_test, y_train, y_test, **kwargs):
    total_x_train = None
    total_y_train = None

    total_x_test = None
    total_y_test = None

    data_crop = kwargs.pop('data_crop',0.5)
    sub_sample = kwargs.pop('sub_sample', 2)
    average_step = kwargs.pop('average_step', 2)
    add_noise = kwargs.pop('add_noise',True)
    verbose = kwargs.pop('verbose',False)
    #Recall X -> [N, C, H]

    #1. Data Cropping
    H = int(data_crop*X_train.shape[2])
    X_train = X_train[:, :, 0:H]
    X_test = X_test[:,:, 0:H]
    if verbose:
        print('Shape of X train after trimming is: {}'.format(X_train.shape))
    
    #2. Maxpool
    X_max_train = np.max(X_train.reshape(X_train.shape[0], X_train.shape[1], -1, sub_sample), axis=3)
    X_max_test = np.max(X_test.reshape(X_test.shape[0], X_test.shape[1], -1, sub_sample), axis=3)
    
    total_x_train = X_max_train
    total_y_train = y_train

    total_x_test = X_max_test
    total_y_test = y_test

    if verbose:
        print('Shape of X train after maxpooling: {}'.format(total_x_train.shape))
        print('Shape of X test after maxpooling: {}'.format(total_x_test.shape))

    #3. Average + Noise
    X_average_train = np.mean(X_train.reshape(X_train.shape[0], X_train.shape[1], -1, average_step),axis=3)
    X_average_test = np.mean(X_test.reshape(X_test.shape[0], X_test.shape[1], -1, average_step),axis=3)

    if add_noise:
        X_average_train = X_average_train + np.random.normal(0.0, 0.5, X_average_train.shape)
        X_average_test = X_average_test + np.random.normal(0.0, 0.5, X_average_test.shape)
    
    total_x_train = np.vstack((total_x_train, X_average_train))
    total_y_train = np.hstack((total_y_train, y_train))

    total_x_test = np.vstack((total_x_test, X_average_test))
    total_y_test = np.hstack((total_y_test, y_test))
    if verbose:
        print('Shape of X train after averaging: {}'.format(total_x_train.shape))
        print('Shape of X test after averaging: {}'.format(total_x_test.shape))
    
    #4. Subsampling
    for i in range(sub_sample):
        
        X_subsample_train = X_train[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X_train[:, :,i::sub_sample].shape) if add_noise else 0.0)
            
        total_x_train = np.vstack((total_x_train, X_subsample_train))
        total_y_train = np.hstack((total_y_train, y_train))

        X_subsample_test = X_test[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X_test[:, :,i::sub_sample].shape) if add_noise else 0.0)
            
        total_x_test = np.vstack((total_x_test, X_subsample_test))
        total_y_test = np.hstack((total_y_test, y_test))
    
    if verbose:
        print('Shape of X train after subsampling and concatenating:',total_x_train.shape)
        print('Shape of Y train after subsampling and concatenating:',total_y_train.shape)
        print('Shape of X test after subsampling and concatenating:',total_x_test.shape)
        print('Shape of Y train after subsampling and concatenating:',total_y_test.shape)
    
    # #5. Reshape into (N,C,H,W)
    # total_x_train = total_x_train.reshape(total_x_train.shape[0], total_x_train.shape[1], total_x_train.shape[2], 1)
    # total_x_test = total_y_test.reshape(total_x_test.shape[0], total_x_test.shape[1], total_x_test.shape[2], 1)

    return total_x_train,total_y_train,total_x_test,total_y_test

#Reshape Data into (N,C,H,W) and put into pytorch Datasets
def Dataset_torch(x,y,transform = None, verbose = False):
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    if verbose:
        print('Shape of x set after adding width info:',x.shape)
    
    dataset = Dataset(x, y, transform=transform)
    x_shape = x.shape
    return x_shape[1:len(x_shape)],dataset