#
# tests go here
#
from variableModel import Model
from dataset import Dataset
import helper as H
import constants as C

import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

use_cuda = T.cuda.is_available()
device = T.device("cuda:0" if use_cuda else "cpu")

# Initialize dataset
dataset = Dataset()
X, y = dataset.preprocess()

# Split and scale the dataset
X_train, X_test, y_train, y_test = dataset.split(X, y, fraction=0.3)
X_train, X_test = dataset.scale(X_train, X_test)

# Load data into PyTorch DataLoader
trainloader, testloader = H.load(X_train, X_test, y_train, y_test)

# Initialize the model for a single agent
mVar = Model(0.01, [12], 2, [8, 6, 6], ['relu', 'relu', 'relu'], trainloader, testloader).to(device)

# Train and test the model
train_acc, train_loss = mVar.train()
test_acc, test_loss = mVar.test()
print("Training Accuracy: ", train_acc, " Training Loss: ", train_loss)
print("Testing Accuracy: ", test_acc, " Testing Loss: ", test_loss)

# Initialize model with new layers
mVar.initialise([8, 6])
mVar = mVar.to(device)

# Train and test the model again
train_acc, train_loss = mVar.train()
test_acc, test_loss = mVar.test()
print("Training Accuracy (after initialization): ", train_acc, " Training Loss: ", train_loss)
print("Testing Accuracy (after initialization): ", test_acc, " Testing Loss: ", test_loss)
