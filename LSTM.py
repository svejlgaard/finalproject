import matplotlib.pyplot as plt
import numpy as np
import os, contextlib, sys
#rom loaddata import loaddata
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import explained_variance_score, r2_score

# Sets the directory to the current directory
os.chdir(sys.path[0])


def loaddata(directory):
    feature_list = list()
    train_data_list = list()
    train_target_list = list()
    for i,filename in enumerate(os.listdir(directory)):
        raw_data = np.genfromtxt(f'{directory}/{filename}', delimiter=',', names=True)
        data_matrix = np.genfromtxt(f'{directory}/{filename}', delimiter=',',skip_header=1)
        
        if i==0: 
            feature_list.append(raw_data.dtype.names)
            train_data_list.append(data_matrix[:,[i for i in range(len(data_matrix[1])) if i!=1]])
            train_target_list.append(data_matrix[:,1])
        elif i==len(os.listdir(directory))-1:
            test_data = data_matrix[:,[i for i in range(len(data_matrix[1])) if i!=1]]
            test_target = data_matrix[:,1]
        else:
            train_data_list.append(data_matrix[:,[i for i in range(len(data_matrix[1])) if i!=1]])
            train_target_list.append(data_matrix[:,1])
    #train_data_list = np.vstack((train_data_list))
    #train_target_list =  np.concatenate(train_target_list)
    return train_data_list, train_target_list, test_data, test_target

def get_explained(model, data, target):
    model.eval()
    pred = model(data).squeeze().detach().numpy()
    target = target.detach().numpy()

    return explained_variance_score(target, pred), r2_score(target, pred)
    

train_data_list, train_target_list, test_data, test_target = loaddata('data')

#Converting to torch format

train_data_torch = [torch.from_numpy(data).unsqueeze(dim=1).float() for data in train_data_list]
train_target_torch = [torch.from_numpy(data).unsqueeze(dim=1).float() for data in train_target_list]

test_data_torch, test_target_torch = torch.from_numpy(np.array(test_data)).unsqueeze(dim=1).float(), torch.from_numpy(np.array(test_target)).float()


class LSTMTagger(nn.Module):

    def __init__(self, inputsize, hiddensize):
        super(LSTMTagger, self).__init__()
        self.inputsize = inputsize
        self.hiddensize = hiddensize

        self.lstm = nn.LSTM(input_size=self.inputsize, hidden_size = self.hiddensize)
        self.hidden2radial = nn.Linear(in_features=hiddensize, out_features=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = F.relu(x)
        x = self.hidden2radial(x)
        return x
    

# Defining epochs

n_epochs = 100

model = LSTMTagger(27,300)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(),lr=0.01)

model.train()
breakpoint()
for e in range(n_epochs):
    epoch_losses = list()
    for batch in range(len(train_data_list)):
        model.zero_grad()
        optimizer.zero_grad() 
        prediction = model(train_data_torch[batch])
        target = train_target_torch[batch]
        # Calculating the loss function
        loss = criterion(prediction.squeeze(dim=2), target)
        epoch_losses.append(float(loss))
        # Calculating the gradient
        loss.backward()
        optimizer.step()
    print(e, np.mean(epoch_losses))

with torch.no_grad():
    for i, traindata in enumerate(train_data_torch):
        print(get_explained(model, traindata, train_target_torch[i] ))
    print(get_explained(model,test_data_torch,test_target_torch))
    