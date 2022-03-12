import enum
from re import X
from sched import scheduler
from unittest import TestLoader
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.autograd import Variable
from sklearn.model_selection import KFold


from CNN import *

'''
Function might required
1. __init__()
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data containing:
          'X_train': Array, shape (N_train, d_1, ..., d_k) of training images
          'X_val': Array, shape (N_val, d_1, ..., d_k) of validation images
          'y_train': Array, shape (N_train,) of labels for training images
          'y_val': Array, shape (N_val,) of labels for validation images

        Optional arguments:
        - update_rule: A string giving the name of an update rule in optim.py.
          Default is 'sgd'.
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters (see optim.py) but all update rules require a
          'learning_rate' parameter so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the
          learning rate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient
          during training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every
          print_every iterations.
        - verbose: Boolean; if set to false then no output will be printed
          during training.
        - K_fold
        ""
    
    Call reset() every initialization

2. _reset()
   -->reset the model & every hypermeters

3. check_accuracy()


1. Training Function
   1.K-cross validation
    -->record loss (for plot)
    -->record train_accuracy, validate_accuracy (for plot)
   2.tqdm (optinonal)
   3. return the trained Model 

   

2. Test Function
   1. check test accuracy
'''

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

# def reset_model(m):
#   for name, module in m.named_children():
#     if hasattr(module, 'reset_parameters'):
#       print('Layer Name: ', name)
#       module.reset_parameters()
  

def train(x_shape, data_set, model_name, criterion, **kwargs):

    
    lr = kwargs.pop('lr', 3e-4)
    k_folds = kwargs.pop('k_folds',5)
    lr_decay = kwargs.pop('lr_decay', True)
    batch_size = kwargs.pop('batch_size', 64)
    num_epochs = kwargs.pop('num_epochs', 50)
    verbose = kwargs.pop('verbose', False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loss_history = defaultdict(list)
    results = defaultdict(float)
    train_acc_history = defaultdict(list)
    valid_acc_history = defaultdict(list)

    torch.manual_seed(42)
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # if lr_decay:
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(data_set)):
        train_subsample = SubsetRandomSampler(train_idx)
        valid_subsample = SubsetRandomSampler(valid_idx)

        if verbose:
          print(f'===============FOLD {fold}===============')
        
        trainloader = DataLoader(data_set, batch_size, sampler = train_subsample)
        validloader = DataLoader(data_set, batch_size, sampler = valid_subsample)

        #Initial model for each fold
        if model_name == 'CNN':
          model = CNN(x_shape).to(device)

        optimizer = torch.optim.Adam(model.parameters(),lr)
        print('Model Initialized')

        for epoch in range(num_epochs):
          avg_loss = 0.0
          num_correct_train = 0
          num_sample_train = 0
          
          for batch_idx, (data, target) in enumerate(trainloader):
            model.train(True)
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
            train_acc_history[fold] = train_acc.item()

            if verbose:
              if batch_idx % 100 == 99 and epoch %10 == 9:
                print('Loss afer %2d epoch: %.3f, with the accuracy %.3f' %((epoch+1), avg_loss/100, train_acc))

            del (data,target)
        
          #Start validation
          num_correct = 0
          num_samples = 0
          with torch.no_grad():
            for i, (x_val, y_val) in enumerate(validloader):
              model.eval()
              outputs = model(x_val)

              _,predict = outputs.max(1)
              num_correct += (predict == y_val).sum()
              num_samples += predict.size(0)
          
              acc = 100.0 * (num_correct/num_samples)
              valid_acc_history[fold] = acc.item()
              if verbose and epoch % 10 == 9 and i == 1:
                print(f'The validation accuracy at epoch {epoch+1} is: {acc:.2f}')
          results[fold] = acc.item()


        #save model weights for each fold, and take average at the end for testing
        if fold != k_folds-1:
          # Save the last model weights
          model_scripted1 = torch.jit.script(model)
          file_name = 'trained_model/model_scripted' + str(fold) +'.pt'
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
    with open('exp_data/loss_history.json', 'w') as f:
      json.dump(loss_history,f)
    
    with open('exp_data/valid_results.json','w') as f:
      json.dump(results,f)
    
    with open('exp_data/train_acc_history.json','w') as f:
      json.dump(train_acc_history,f)
    
    with open('exp_data/valid_acc_history.json','w') as f:
      json.dump(valid_acc_history,f)
    
    model_scripted1 = torch.jit.script(model)
    file_name = 'trained_model/model_scripted' + str(fold) +'.pt'
    model_scripted1.save(file_name)

    torch.cuda.empty_cache()
    return model

