import torch
import torch.nn as nn
import torch.nn.functional as F


# 2-layer fc
class MNISTFull(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        h1 = config['h1']
        self.fc1 = torch.nn.Linear(28*28, h1)
        self.fc2 = torch.nn.Linear(h1, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        # x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return x


# Softmax classifier
# class MNISTFull(torch.nn.Module):

#     def __init__(self, config):
#         super().__init__()
#         self.fc = torch.nn.Linear(28*28, 10)

#     def forward(self, x):
#         x = x.view(-1, 28 * 28)
#         x = self.fc(x)
#         return x


class MNISTSharedLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        h1 = config['h1']
        self.fc1 = torch.nn.Linear(28*28, h1)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        return x


class MNISTPersonalLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        h1 = config['h1']
        self.fc2 = torch.nn.Linear(h1, 10)
    
    def forward(self, x):
        x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return x