import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from CNN import *
# from tqdm import tqdm

X_test = np.load("project_data\X_test.npy")
y_test = np.load("project_data\y_test.npy")
person_train_valid = np.load("project_data\person_train_valid.npy")
X_train_valid = np.load("project_data\X_train_valid.npy")
y_train_valid = np.load("project_data\y_train_valid.npy")
person_test = np.load("project_data\person_test.npy")

y_train_valid -= 769
y_test -= 769

ind_valid = np.random.choice(2115, 500, replace=False)
ind_train = np.array(list(set(range(2115)).difference(set(ind_valid))))

# Creating the training and validation sets using the generated indices
(x_train, x_valid) = X_train_valid[ind_train], X_train_valid[ind_valid] 
(y_train, y_valid) = y_train_valid[ind_train], y_train_valid[ind_valid]

'''
Shape of training set after adding width info: (1615, 22, 1000, 1)
Shape of validation set after adding width info: (500, 22, 1000, 1)
Shape of test set after adding width info: (443, 22, 1000, 1)
Shape of training set after dimension reshaping: (1615, 1000, 1, 22)
Shape of validation set after dimension reshaping: (500, 1000, 1, 22)
Shape of test set after dimension reshaping: (443, 1000, 1, 22)
'''


print('Shape of training labels:',y_train.shape)
print('Shape of validation labels:',y_valid.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_train.shape[2], 1)
x_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
print('Shape of training set after adding width info:',x_train.shape)
print('Shape of validation set after adding width info:',x_valid.shape)
print('Shape of test set after adding width info:',x_test.shape)


# x_train = np.swapaxes(x_train, 1,3)
# x_train = np.swapaxes(x_train, 1,2)
# x_valid = np.swapaxes(x_valid, 1,3)
# x_valid = np.swapaxes(x_valid, 1,2)
# x_test = np.swapaxes(x_test, 1,3)
# x_test = np.swapaxes(x_test, 1,2)
# print('Shape of training set after dimension reshaping:',x_train.shape)
# print('Shape of validation set after dimension reshaping:',x_valid.shape)
# print('Shape of test set after dimension reshaping:',x_test.shape)

######################Training########################
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

x_train = torch.from_numpy(x_train)
x_train = x_train.type(torch.float32).to(device=device)
y_train = torch.from_numpy(y_train)
y_train = y_train.type(torch.LongTensor).to(device=device)


def train(x,y,num_epoch,model):
    for epoch in range(num_epoch):
        num_correct = 0
        num_samples = 0
        scores = model(x)
        loss = criterion(scores, y)

        _,predictions = scores.max(1)
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if epoch % 100 == 99:
            print(f'At Epoch: {epoch}, Got {num_correct} / {num_samples} correct, with accuracy {float(num_correct) / float(num_samples)*100:.2f}')

def check_accuracy(x_valid,y_valid,model):
    num_correct = 0
    num_samples = 0
    acc = 0
    model.eval()

    with torch.no_grad():
        x_valid = torch.from_numpy(x_valid)
        x_valid = x_valid.type(torch.float32).to(device=device)
        y_valid = torch.from_numpy(y_valid)
        y_valid = y_valid.type(torch.LongTensor).to(device=device)

        scores = model(x_valid)
        _,predictions = scores.max(1)
        num_correct += (predictions == y_valid).sum()
        num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples)*100:.2f}')
    
    model.train()
    acc = float(num_correct) / float(num_samples)*100
    return acc

train(x_train,y_train,2000,model)
check_accuracy(x_valid,y_valid,model)
check_accuracy(x_test,y_test,model)

## Test Accuracy: 71.78%