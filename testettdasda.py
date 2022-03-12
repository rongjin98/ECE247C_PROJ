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
x_test_shape,test_dataset = Dataset_torch(X_test,y_test,verbose=False)

test_loader = DataLoader(test_dataset, len(test_dataset))


def reset_model(m):
  for layer in m.children():
    if hasattr(layer, 'reset_parameters'):
      layer.reset_parameters()

model = nn.Linear(3,3)
lr = 3e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
sth = optimizer.state_dict

model1 = torch.jit.load('model_scripted1.pt')
model2 = torch.jit.load('averaged_model.pt')

def check_accuracy(x_valid,y_valid,model):
    num_correct = 0
    num_samples = 0
    acc = 0
    model.eval()

    with torch.no_grad():
        # x_valid = torch.from_numpy(x_valid)
        # x_valid = x_valid.type(torch.float32).to(device=device)
        # y_valid = torch.from_numpy(y_valid) 
        # y_valid = y_valid.type(torch.LongTensor).to(device=device)

        scores = model(x_valid)
        _,predictions = scores.max(1)
        num_correct += (predictions == y_valid).sum()
        num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples)*100:.2f}')
    
    model.train()
    acc = float(num_correct) / float(num_samples)*100
    return acc

for x,y in test_loader:
    check_accuracy(x,y,model1)
    check_accuracy(x,y,model2)