from data_loader import *
import torch
from torch.utils.data import Dataset, DataLoader
from CNN import *
import torch.nn as nn
import torch.optim as optim

path = "project_data/"
X_train_valid, y_train_valid, X_test, y_test = data_loader(path,True)
X_train_valid, y_train_valid, X_test, y_test = data_process(X_train_valid, 
        X_test, y_train_valid, y_test, verbose = True)
x_shape,train_dataset = Dataset_torch(X_train_valid,y_train_valid,verbose=True)
x_test_shape,test_dataset = Dataset_torch(X_test,y_test,verbose=True)

train_loader = DataLoader(train_dataset,batch_size=64)
test_loader = DataLoader(test_dataset, len(test_dataset))

num_epoch = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
model = CNN(x_shape).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epoch):
   for batch_idx, (data, targets) in enumerate(train_loader):
        num_correct = 0
        num_samples = 0
        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)
        loss = criterion(scores, targets)

        _,predictions = scores.max(1)
        num_correct += (predictions == targets).sum()
        num_samples += predictions.size(0)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if epoch % 10 == 9 and batch_idx == 60:
            print(f'At Epoch: {epoch}, Got {num_correct} / {num_samples} correct, with accuracy {float(num_correct) / float(num_samples)*100:.2f}')

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
    check_accuracy(x,y,model)