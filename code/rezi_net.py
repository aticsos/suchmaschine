import torch
import torch.nn as nn
class rezi_net(nn.Module):
    def __init__(self):
        super(rezi_net, self).__init__()
        self.l_in = nn.Linear(192, 100)
        self.l_1 = nn.Linear(100, 50)
        self.l_out = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.tanh(self.l_in(x))
        x = torch.tanh(self.l_1(x))
        x = self.l_out(x)

        return x