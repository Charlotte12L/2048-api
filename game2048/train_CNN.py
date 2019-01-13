import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from Model import Net
import time
import pandas as pd
import numpy as np
import csv



batch_size = 128
NUM_EPOCHS = 80

#loading data
csv_data = pd.read_csv('./direct/new/game2048/all.csv')
csv_data = csv_data.values
board_data = csv_data[:,0:16]
X = np.int64(board_data)
X = np.reshape(X, (-1,4,4))
direction_data = csv_data[:,16]
Y = np.int64(direction_data)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,shuffle=False)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

train_dataset = torch.utils.data.TensorDataset(X_train,Y_train)
test_dataset = torch.utils.data.TensorDataset(X_test,Y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True
)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False
)

model = Net()
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        data = data.unsqueeze(dim=1)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    torch.save(model.state_dict(), './direct/new/game2048/saved/epoch_{}.pkl'.format(epoch))

def test(epoch):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        with torch.no_grad():
            data = Variable(data).cuda()
            target =Variable(target).cuda()
        data = data.unsqueeze(dim=1)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))


for epoch in range(40,41):
    model.train()
    train(epoch)
    model.eval()
    test(epoch)
