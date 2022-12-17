import numpy as np
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, input_shape) -> None:
        super().__init__()
        self.input_shape = input_shape
        if len(input_shape) == 2:
            input_shape = (1, *input_shape)
        if len(input_shape) == 3:
            self.nn = nn.Sequential(
                nn.Conv2d(input_shape[0], 64, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=1),
                nn.ReLU()
            )
        elif len(input_shape) == 1:
            self.nn = nn.Sequential(
                nn.Linear(input_shape[0], 64),
                nn.ReLU()
            )

        x = torch.zeros(input_shape)
        x = x.unsqueeze(0)
        x = self.nn(x)
        self.num_feature = np.prod(x.shape[1:])
        print('CNN features:', self.num_feature)

    def forward(self, x):
        return self.nn(x)


class VDNet(nn.Module):
    def __init__(self, num_feature, num_act) -> None:
        super().__init__()
        self.vnet = nn.Sequential(
            nn.Linear(num_feature, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.dnet = nn.Sequential(
            nn.Linear(num_feature, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, num_act)
        )

    def forward(self, x):
        v = self.vnet(x)
        d = self.dnet(x)
        return d+v-d.mean()
        # return d


class DuelingNetwork(nn.Module):
    def __init__(self, input_shape, num_act) -> None:
        super().__init__()
        self.features = CNN(input_shape)
        num_feature = self.features.num_feature
        self.vdnet = VDNet(num_feature, num_act)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.vdnet(x)
        return x


if __name__ == '__main__':
    dn = DuelingNetwork((1, 16, 16), 2)
    x = torch.rand(1, 1, 16, 16)
    print(dn.forward(x))
