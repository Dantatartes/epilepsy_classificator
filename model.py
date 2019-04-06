import torch
from torch.utils import data
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import os

import time


class Dataset(data.Dataset):
    def __init__(self, path):
        self.path = path

    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, idx):
        with open(f'{self.path}/seiz_{idx}.pkl', 'rb') as f:
            data = pickle.load(f)
            return {'data': data[1], 'label': labels_2_num[data[0]]}


class Model(nn.Module):
    def __init__(self, D_out):
        super(Model, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=2)

        self.fc1_1 = nn.Linear(in_features=900, out_features=256)
        self.fc1_2 = nn.Linear(in_features=256, out_features=64)
        self.fc1_3 = nn.Linear(in_features=64, out_features=32)
        self.fc1_4 = nn.Linear(in_features=32, out_features=D_out)

    def forward(self, x):

        h = F.relu(self.fc1_1(x))
        h = F.relu(self.fc1_2(h))
        h = F.relu(self.fc1_3(h))
        h = F.softmax(self.fc1_4(h))

        return h
