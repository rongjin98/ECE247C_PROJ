import enum
from re import X
from sched import scheduler
from unicodedata import bidirectional
from unittest import TestLoader
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict

import torch
import os
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.autograd import Variable
from sklearn.model_selection import KFold


from CNN import *
from CRNN import CNN_GRU, CNN_LSTM
from data_loader import *

def check_accuracy(x_valid,y_valid,model,criterion):
    num_correct = 0
    num_samples = 0
    acc = 0
    model.eval()

    with torch.no_grad():
        scores = model(x_valid)
        loss = criterion(scores, y_valid)
        
        _,predictions = scores.max(1)
        num_correct += (predictions == y_valid).sum()
        num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples)*100:.2f}')
    
    model.train()
    acc = float(num_correct) / float(num_samples)*100
    return loss,acc
  

def train(x_shape, data_set, model_name, criterion, **kwargs):

    
    lr = kwargs.pop('lr', 3e-4)
    k_folds = kwargs.pop('k_folds',5)
    lr_decay = kwargs.pop('lr_decay', True)
    batch_size = kwargs.pop('batch_size', 64)
    num_epochs = kwargs.pop('num_epochs', 50)
    verbose = kwargs.pop('verbose', False)
    subset_sampler = kwargs.pop('subsampler', True)

    dropout = kwargs.pop('dropout', 0.5)
    bidirection = kwargs.pop('birdiection', True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    loss_history = defaultdict(list)
    results = defaultdict(float)
    train_acc_history = defaultdict(list)
    valid_acc_history = defaultdict(list)
    valid_loss_history = defaultdict(list)
    

    torch.manual_seed(42)
    kfold = KFold(n_splits=k_folds, shuffle=True)

    #Path for Data Saving
    data_path = model_name + '_exp_data'
    model_path = model_name + '_trained_model'
    if os.path.exists(data_path) == False:
      os.makedirs(data_path)
    
    if os.path.exists(model_path) == False:
      os.makedirs(model_path)
    
    
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(data_set)):
        
        if subset_sampler:
          train_subsample = SubsetRandomSampler(train_idx)
          valid_subsample = SubsetRandomSampler(valid_idx)
          trainloader = DataLoader(data_set, batch_size, sampler = train_subsample)
          validloader = DataLoader(data_set, batch_size, sampler = valid_subsample)
        else:
          x_train,y_train = data_set.__getitem__(train_idx)
          x_valid,y_valid = data_set.__getitem__(valid_idx)
        
          train_set = Dataset(x_train, y_train, transform=None)
          valid_set = Dataset(x_valid, y_valid, transform=None)
          trainloader = DataLoader(train_set, batch_size)
          validloader = DataLoader(valid_set, batch_size)
          

        if verbose:
          print(f'===============FOLD {fold}===============')

        #Initial model for each fold
        if model_name == 'CNN':
          model = CNN(x_shape, dropout).to(device)
        elif model_name == 'LSTM':
          model = CNN_LSTM(x_shape, dropout, bidirection).to(device)
        elif model_name == 'GRU':
          model = CNN_GRU(x_shape, dropout, bidirection).to(device)

        optimizer = torch.optim.Adam(model.parameters(),lr)
        if lr_decay:
          scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        print('Model Initialized')

        for epoch in range(num_epochs):
          avg_loss = 0.0
          num_correct_train = 0
          num_sample_train = 0
          model.train(True)

          for batch_idx, (data, target) in enumerate(trainloader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            scores = model(data)

            loss = criterion(scores, target)
            
            avg_loss += loss.item()
            loss_history[fold].append(loss.item())

            loss.backward()
            optimizer.step()

            #check accuracy
            _,pred_train = scores.max(1)
            num_correct_train += (pred_train == target).sum()
            num_sample_train += pred_train.size(0)

            train_acc = 100.0* (num_correct_train/num_sample_train)
            train_acc_history[fold].append(train_acc.item())

            if verbose:
              if batch_idx % 100 == 99 and epoch %10 == 9 :
                print('Loss afer %2d epoch: %.3f, with the accuracy %.3f' %((epoch+1), avg_loss/100, train_acc))  
            del (data,target)

          #Because of high correlation, valid accuracy is not typically useful
          if lr_decay:
            scheduler.step(avg_loss)
        
          #Start validation
          num_correct = 0
          num_samples = 0
          num_batch = 0
          model.eval()

          with torch.no_grad():
            for i, (x_val, y_val) in enumerate(validloader):
              num_batch += 1
              outputs = model(x_val)
              valid_loss = criterion(outputs,y_val)
              valid_loss_history[fold].append(valid_loss.item())

              _,predict = outputs.max(1)
              num_correct += (predict == y_val).sum()
              num_samples += predict.size(0)
          
              acc = 100.0 * (num_correct/num_samples)
              valid_acc_history[fold].append(acc.item())
              if verbose and epoch % 10 == 9 and i == 1:
                print(f'The validation accuracy at epoch {epoch+1} is: {acc:.2f}')

          
          results[fold] = acc.item()


        #save model weights for each fold, and take average at the end for testing
        if fold != k_folds-1:
          # Save the last model weights
          model_scripted1 = torch.jit.script(model)
          file_name = model_path + '/model_scripted' + str(fold) +'.pt'
          model_scripted1.save(file_name)
          del(model,optimizer)
          print(f'Deleted mode')
    
    print(f'All {kfold} Folds Completed')

    # Print validation accuracy for each fold, and the average accuracy
    sum_acc = 0.0
    for key in results:
      print(f'Fold {key}: {results[key]} %')
      sum_acc += results[key]
    print('Average is: ', sum_acc/len(results.items()), '%')

    # Dump loss history and result history to jason
    print('Saving training results...')
    
    
    with open((data_path+'/loss_history.json'), 'w') as f:
      json.dump(loss_history,f)
    
    with open((data_path+'/valid_results.json'),'w') as f:
      json.dump(results,f)
    
    with open((data_path+'/train_acc_history.json'),'w') as f:
      json.dump(train_acc_history,f)
    
    with open((data_path+'/valid_acc_history.json'),'w') as f:
      json.dump(valid_acc_history,f)
    
    with open((data_path+'/valid_loss_history.json'),'w') as f:
      json.dump(valid_loss_history,f)
    
    model_scripted1 = torch.jit.script(model)
    file_name = model_path + '/model_scripted' + str(fold) +'.pt'
    model_scripted1.save(file_name)

    torch.cuda.empty_cache()
    return model

def test(test_dataset, model, verbose=True):
  num_correct = 0
  num_samples = 0
  model.eval()
  test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
  with torch.no_grad():
    for (x_val, y_val) in test_loader:
      outputs = model(x_val)
      _,predict = outputs.max(1)
      num_correct += (predict == y_val).sum()
      num_samples += predict.size(0)
          
      acc = 100.0 * (num_correct/num_samples)
      if verbose:
        print(f'The test accuracy is: {acc:.2f} %')
  return acc