import torch
import torch.nn as nn
import torch.nn.functional as F


class EvenNet(nn.Module):

    def __init__(self):
        super(EvenNet, self).__init__()
        self.fc1 = nn.Linear(1000, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = EvenNet()