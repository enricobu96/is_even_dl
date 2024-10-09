import torch
import torch.nn as nn
import torch.nn.functional as F

PI = torch.acos(torch.zeros(1)).item() * 2

cf = lambda x: torch.cos(PI*x-PI)/2+0.5

def heaviside(x):
    x = torch.where(x>=0.5, 1, x)
    x = torch.where(x<0.5, 0, x)
    # x = torch.where(x==0, 0, x)
    return x
    print(x)
    exit()

    aux = []
    for i in x:
        if i.item() < 0:
            aux.append(-1)
        elif i.item() == 0:
            aux.append(0)
        else:
            aux.append(1)
    return torch.tensor(aux)

class EvenNet(nn.Module):

    def __init__(self):
        super(EvenNet, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 3)
        self.fc3 = nn.Linear(3, 1)
        self.act = F.relu
        self.sigm = F.sigmoid

    def forward(self, x):
        # print(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigm(x)
        # print(x)
        # quit()
        # x = heaviside(x)
        # x = x if x==0 else 1
        return x