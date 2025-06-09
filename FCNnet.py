from torch import nn

class FullyConnectedNet(nn.Module):
    """全连接神经网络模型"""
    def __init__(self,input_size=784,num_classes=10):
        super(FullyConnectedNet, self).__init__()
        self.net  = nn.Sequential(nn.Flatten(), nn.Linear(input_size, num_classes))
    def forward(self, x):
        return self.net(x)