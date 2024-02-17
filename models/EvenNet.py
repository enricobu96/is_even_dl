import torch
import torch.nn as nn
import torch.nn.functional as F

PI = torch.acos(torch.zeros(1)).item() * 2

cf = lambda x: torch.cos(PI*x-PI)/2+0.5

def heaviside(x):
    x = torch.where(x>0, 1, x)
    x = torch.where(x<0, -1, x)
    x = torch.where(x==0, 0, x)
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

    def __init__(self, input_dim):
        super(EvenNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim*4)
        self.fc2 = nn.Linear(input_dim*4, input_dim*2)
        self.fc3 = nn.Linear(input_dim*2, input_dim)
        self.fc4 = nn.Linear(input_dim, input_dim)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = heaviside(x)
        x = F.relu(self.fc4(x))
        return x