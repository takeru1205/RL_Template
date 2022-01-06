import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    import torch
    input_tensor = torch.rand(8, 16)
    net = Net(state_dim=16, act_dim=4)
    output_tensor = net(input_tensor)
    assert output_tensor.shape == (8, 4)

