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
from train import train
from test import test


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



for epoch in range(40, NUM_EPOCHS):
    model.train()
    train(epoch)
    model.eval()
    test(epoch)
