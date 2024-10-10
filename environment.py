import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from dataset import Dataset
import constants as C
import helper as H
from variableModel import Model

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

class Environment():
    def __init__(self, path='churn_modelling.csv'):
        self.path = path
        self.dataset = Dataset(path=self.path)
        self.X, self.y = self.dataset.preprocess()
        self.X_train, self.X_test, self.y_train, self.y_test = self.dataset.split(self.X, self.y, fraction=0.2)
        self.X_train, self.X_test = self.dataset.scale(self.X_train, self.X_test)
        self.train_loader, self.test_loader = H.load(self.X_train, self.X_test, self.y_train, self.y_test)

        self.input_dims = [self.X.shape[1]]
        self.output_dims = len(np.unique(self.y))
        print("Dims of X_train is {}".format(H.get_dimensions(data=self.X_train)))
        print("Dims of y_train is {}".format(H.get_dimensions(data=self.y_train)))
        print("Input dims is {}, output dims is {}".format(self.input_dims, self.output_dims))

        self.agent_state = np.random.randint(C.MIN_NODES, C.MAX_NODES)
        self.func = 'relu'
        self.activation = self.func
        
        print("Environment initialized ...")
        self.model = Model(0.01, self.input_dims, self.output_dims, self.agent_state, [self.activation], self.train_loader, self.test_loader)
        self.model = self.model.to(device)

    def reset(self):
        self.agent_state = np.random.randint(C.MIN_NODES, C.MAX_NODES)
        self.model.initialise(self.agent_state)
        self.model = self.model.to(device)
        return self.agent_state

    def sample_reset(self, state_passed):
        self.agent_state = state_passed
        self.model.initialise(self.agent_state)
        self.model = self.model.to(device)
        return state_passed

    def step(self, action):
        state_, reward = self.change_neurons(action)
        return (state_, reward)

    def change_neurons(self, action):
        current_nodes = self.model.hidden_layer_array[0]  # Single agent, so index 0
        if action > 0:
            next_state = self.model.add_neurons(int(action), 0)
        else:
            next_state = self.model.remove_neurons(-int(action), 0)

        self.model = self.model.to(device)
        train_acc, train_loss = self.model.train()
        test_acc, test_loss = self.model.test()
        reward = H.reward(train_acc, train_loss,
                          test_acc, test_loss,
                          next_state, 0,  # Single agent, so agent_no = 0
                          self.X.shape[1],
                          self.output_dims,
                          action, current_nodes)

        print("Train_acc : ", train_acc)
        print("Test_acc : ", test_acc)
        print("Train_loss : ", train_loss)
        print("Test_loss : ", test_loss)

        return (next_state, reward)

    def seed(self):
        pass
