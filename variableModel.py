import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import constants as C

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self, lr, input_dims, output_dims, hidden_layer_array, non_linear_functions, trainloader, testloader): 
        super(Model, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_layer_array = hidden_layer_array
        self.non_linear_functions = non_linear_functions
        self.num_layers = len(hidden_layer_array)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.trainloader = trainloader
        self.testloader = testloader
        self.initialise(hidden_layer_array)

    def initialise(self, hidden_layer_array):
        self.hidden_layer_array = hidden_layer_array
        self.num_layers = len(hidden_layer_array)
        self.fcs = nn.ModuleList([nn.Linear(self.input_dims[0], hidden_layer_array[0])])

        if self.num_layers > 1:
            hidden_layers = zip(hidden_layer_array[:-1], hidden_layer_array[1:])
            self.fcs.extend([nn.Linear(h1, h2) for h1, h2 in hidden_layers])

        self.output = nn.Linear(hidden_layer_array[-1], self.output_dims)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.fcs[i](x)
            x = getattr(F, self.non_linear_functions[i])(x)
        x = self.output(x)  
        return x

    def print_param(self):
        x = next(self.parameters()).data
        print(x)
    
    def train(self):
        loss_list, acc_list = [], []
        for epochs in range(C.EPOCHS):
            correct = 0
            total = 0
            train_loss = 0
            for data, target in self.trainloader:
                data, target = data.to(device), target.to(device)

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

        return acc_list[-1], loss_list[-1]

    def test(self):
        correct = 0
        total = 0
        val_loss = 0
        with T.no_grad():
            for data, target in self.testloader:
                data, target = data.to(device), target.to(device)
                output = self.forward(data.float())
                loss = self.criterion(output, target.squeeze())
                val_loss += loss.item() * data.size(0)

                _, predicted = T.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target.squeeze()).sum().item()

        val_loss = val_loss / len(self.testloader.dataset)
        accuracy = 100 * correct / float(total)
        return accuracy, val_loss
