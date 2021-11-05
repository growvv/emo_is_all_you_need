import torch
import torch.nn as nn

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        assert x.size(0) == y.size(0)
        len = x.size(0)
        loss = 0
        for i in range(len):
            loss += (x[i]-y[i])*(x[i]-y[i])
        return loss/len

# 多交叉损失函数
def multi_category_focal_loss2(pred, target, gamma=2, alpha=0.25):
    """
    :param pred: 预测值
    :param target: 真实值
    :param gamma:
    :param alpha:
    :return:
    """
    pred = pred.view(-1, 2)
    target = target.view(-1, 1)
    target = target.type(torch.FloatTensor)
    # target = target.type(torch.cuda.FloatTensor)
    eps = 1e-7
    t = 1.0 - target
    pt = (1.0 - pred) * t + eps
    weight = alpha * t + (1.0 - alpha) * (1.0 - t)
    weight = weight * t + eps
    weight = weight * t + eps
    loss = weight * (torch.pow((1.0 - pt), gamma)) * torch.log(pt)
    loss = -1 * loss.sum(1)
    return loss.mean()




if __name__ == "__main__":
    rmse = RMSELoss()
    # x = torch.randint(0,3, (64, 4))
    # y = torch.randn((64, 1))
    x = torch.Tensor([0, 0, 2, 0])
    y = torch.Tensor([0, 1, 0, 0])
    loss = rmse(x, y)
    print(loss)