from data_loader import *
import torch
from torch.utils.data import Dataset, DataLoader
from CNN import *
from train_test import *
import torch.nn as nn
import torch.optim as optim

path = "project_data/"
X_train_valid, y_train_valid, X_test, y_test = data_loader(path,False)
X_train_valid, y_train_valid, X_test, y_test = data_process(X_train_valid, 
        X_test, y_train_valid, y_test, verbose = False)
x_shape,train_dataset = Dataset_torch(X_train_valid,y_train_valid,verbose=False)
x_test_shape,test_dataset = Dataset_torch(X_test,y_test,verbose=False)


train_loader = DataLoader(train_dataset,batch_size=64)
test_loader = DataLoader(test_dataset, len(test_dataset))


num_epoch = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
criterion = nn.CrossEntropyLoss()

model = train(x_shape, train_dataset, 'LSTM', criterion, k_folds = 5,num_epochs = 100, dropout = 0.6, subsampler = True, lr_decay=True, verbose = True, bidirection = False)

def check_accuracy(x_valid,y_valid,model):
    num_correct = 0
    num_samples = 0
    acc = 0
    model.eval()

    with torch.no_grad():
        scores = model(x_valid)
        _,predictions = scores.max(1)
        num_correct += (predictions == y_valid).sum()
        num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples)*100:.2f}')
    
    model.train()
    acc = float(num_correct) / float(num_samples)*100
    return acc

acc = test(test_dataset, model)

