import ipdb
import torch
import torch.nn as nn
import ipdb


class ClassiLoss(nn.Module):
    def __init__(self):
        super(ClassiLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        batch_size = x.size(0)
        loss = 0
        for i in range(batch_size):
            # ipdb.set_trace()
            loss += self.criterion(x[i, :, :], y[i, :])
        
        return loss
