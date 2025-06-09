import torch
import torch.nn as nn



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu=nn.ReLU()
        self.linear=nn.Linear(64*7*7,10)

    def forward(self,x):
        x = self.conv1(x)
        x=self.pool1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.pool2(x)
        x=self.relu(x)
        x=x.view(-1,64*7*7)
        x=self.linear(x)
        return x


