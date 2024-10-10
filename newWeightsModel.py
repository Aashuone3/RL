import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import constants as C
from statistics import mean

class SingleAgentModel(nn.Module):
    def __init__(self, input_dims, output_dims, lr, num_layers, num_nodes, trainloader, testloader):
        super(SingleAgentModel, self).__init__()
        self.input_dims = input_dims
        self.lr = lr
        self.num_layers = num_layers
        self.output_dims = output_dims
        self.num_nodes = num_nodes
        self.fcs = None
        self.output = None
        self.initialise(num_layers, num_nodes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.trainloader = trainloader
        self.testloader = testloader
        
    def initialise(self, layers, neurons):
        nodes = [neurons] * layers
        hidden_layers = zip(nodes[:-1], nodes[1:])
        self.fcs = nn.ModuleList([nn.Linear(self.input_dims, neurons)])
        self.fcs.extend([nn.Linear(h1, h2) for h1, h2 in hidden_layers])
        self.output = nn.Linear(neurons, self.output_dims)
    
    def forward(self, x):
        for layer in self.fcs:
            x = layer(x)
            x = F.relu(x)
        x = self.output(x)
        return x

    def add_neurons(self, num):
        self.num_nodes += num
        self.initialise(self.num_layers, self.num_nodes)
        return [self.num_layers, self.num_nodes]

    def remove_neurons(self, num):
        fin_neurons = max(self.num_nodes - num, 1)
        self.num_nodes = fin_neurons
        self.initialise(self.num_layers, self.num_nodes)
        return [self.num_layers, self.num_nodes]

    def add_layers(self, num):
        self.num_layers += num
        self.initialise(self.num_layers, self.num_nodes)
        return [self.num_layers, self.num_nodes]

    def remove_layers(self, num):
        self.num_layers = max(self.num_layers - num, 1)
        return [self.num_layers, self.num_nodes]

    def print_param(self):
        x = next(self.parameters()).data
        print(x)

    def train(self):
        loss_list, acc_list = [], []
        for epochs in range(C.EPOCHS):
            correct = 0
            total = 0
            train_loss = 0
            loader = self.trainloader
            for data, target in loader:
                self.optimizer.zero_grad()
                output = self.forward(data.float())
                loss = self.criterion(output, target.long().squeeze())
                train_loss += loss.item() * data.size(0)
                loss.backward()
                self.optimizer.step()
                total += target.size(0)
                _, predicted = T.max(output.data, 1)
                correct += (predicted == target.squeeze()).sum().item()
            
            acc_list.append(100 * correct / total)
            loss_list.append(train_loss / total)
            print("Epoch {} / {}: Accuracy is {}, loss is {}".format(epochs, C.EPOCHS, 100 * correct / total, train_loss / total))
        return acc_list[-1], loss_list[-1]
    
    def test(self):
        correct = 0
        total = 0
        val_loss = 0
        with T.no_grad():
            for data, target in self.testloader:
                output = self.forward(data.float())
                loss = self.criterion(output, target.squeeze())
                val_loss += loss.item() * data.size(0)
                _, predicted = T.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target.squeeze()).sum().item()

        val_loss = val_loss / len(self.testloader.dataset)
        accuracy = 100 * correct / float(total)
        return accuracy, val_loss

# Usage example:
# trainloader = your_train_dataloader_here
# testloader = your_test_dataloader_here
# model = SingleAgentModel(input_dims, output_dims, lr, num_layers, num_nodes, trainloader, testloader)
# accuracy, loss = model.train()
# test_accuracy, test_loss = model.test()
