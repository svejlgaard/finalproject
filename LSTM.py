import matplotlib.pyplot as plt
import numpy as np
import os, contextlib, sys
#rom loaddata import loaddata
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import explained_variance_score, r2_score
from bayes_opt import BayesianOptimization

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

    def __init__(self, inputsize, layers, hiddensize):
        super(LSTMTagger, self).__init__()
        self.inputsize = inputsize
        self.hiddensize = hiddensize
        self.layers = layers

        self.lstm = nn.LSTM(input_size=self.inputsize, hidden_size = self.hiddensize, num_layers=layers)
        self.hidden2radial = nn.Linear(in_features=hiddensize, out_features=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = F.relu(x)
        x = self.hidden2radial(x)
        return x
    

# Defining epochs

n_epochs = 100

# Baysian Optimization

def NN_CrossValidation(hiddensize, layers, learning_rate, data, targets):
    estimator = LSTMTagger(27, hiddensize=hiddensize,layers=layers)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=learning_rate)


    n_epochs = 100
    criterion = nn.MSELoss()
    estimator.train()

    for e in range(n_epochs):
        epoch_losses = list()
        #for batch in range(len(train_data_list)):
        estimator.zero_grad()
        optimizer.zero_grad() 
        prediction = estimator(data)
        target = targets
        # Calculating the loss function
        loss = criterion(prediction,target)
        epoch_losses.append(float(loss))
        # Calculating the gradient
        loss.backward()
        optimizer.step()
        #print(e, np.mean(epoch_losses))
    with torch.no_grad():
        estimator.eval()

        train_prediction = estimator(data).squeeze(dim=1)
        print(train_prediction.size())
        #rmse = torch.mean((train_prediction-targets)**2).sqrt()
        #acc_train = torch.mean((train_prediction == targets).float())


        acc_train = explained_variance_score(targets, train_prediction)
        #cval = cross_val_score(estimator, data, targets, scoring='accuracy', cv=5)
        
        return acc_train

def optimize_NN(data, targets, pars, n_iter=10):
    def crossval_wrapper(hiddensize, layers, learning_rate):

        return NN_CrossValidation(hiddensize=int(hiddensize), layers=int(layers), learning_rate=learning_rate,
                                            data=data, 
                                            targets=targets)

    optimizer = BayesianOptimization(f=crossval_wrapper, 
                                     pbounds=pars, 
                                     random_state=27, 
                                     verbose=2)
    optimizer.maximize(init_points=5, n_iter=n_iter)

    return optimizer

parameters_BO = {"hiddensize": (10,300), "learning_rate": (0.1,0.001),"layers": (1,4)}

optimization_data = train_data_torch[1]
optimization_target = train_target_torch[1]

print(optimization_data.size(), optimization_target.size())

BayesianOptimization = optimize_NN(optimization_data,optimization_target,parameters_BO,n_iter=5)

print(BayesianOptimization.max)

params = BayesianOptimization.max['params']



for key, val in params.items():
    if key == 'hiddensize':
        params[key] = int(val)
    if key =='learning_rate':
        params[key] = val
    if key == 'layers':
        params[key] = int(val)




model = LSTMTagger(27,params['hiddensize'],params['layers'])

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(),lr=params['learning_rate'])

model.train()
for e in range(n_epochs):
    epoch_losses = list()
    epoch_evs = list()
    for batch in range(len(train_data_list)):
        model.zero_grad()
        optimizer.zero_grad() 
        prediction = model(train_data_torch[batch])
        target = train_target_torch[batch]
        # Calculating the loss function
        loss = criterion(prediction.squeeze(dim=2), target)
        epoch_losses.append(float(loss))
        evs = explained_variance_score(target.squeeze(dim=1).detach().numpy(),prediction.squeeze(dim=1).detach().numpy())
        epoch_evs.append(evs)
        # Calculating the gradient
        loss.backward()
        optimizer.step()
    print(e, np.mean(epoch_losses),np.mean(epoch_evs))

with torch.no_grad():
    for i, traindata in enumerate(train_data_torch):
        print(get_explained(model, traindata, train_target_torch[i] ))
    print(get_explained(model,test_data_torch,test_target_torch))
    