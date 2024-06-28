import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,in_channels=50):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_channels, 35)
        self.fc2 = nn.Linear(35, 20)
        self.fc3 = nn.Linear(20, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # CrossEntropyLoss includes softmax
        return x