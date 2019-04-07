import torch
from torch import nn
from torch.utils import data as data_utils
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import pickle
import os
import functools
import shutil

import time

labels = ['ABSZ', 'CPSZ', 'FNSZ', 'GNSZ', 'SPSZ', 'TCSZ', 'TNSZ']
labels_2_num = {labels[i]: i for i in range(len(labels))}


class Dataset(data_utils.Dataset):
    def __init__(self, path):
        self.path = path

    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, idx):
        with open(f'{self.path}/seiz_{idx}.pkl', 'rb') as f:
            data = pickle.load(f)
            return {'data': data[1], 'label': labels_2_num[data[0]]}


class FcModelNN(nn.Module):
    def __init__(self, D_out):
        super(FcModelNN, self).__init__()
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

def train_model(model, criterion, optimizer, dataloader_, is_cuda=False, num_epochs=25):
    since = time.time()
    
    losses = []
    model.train(True)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0

        for data in dataloader_:
            inputs, labels = data['data'].float(), data['label']
            
            if is_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            
            loss = criterion(model(inputs), labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader_.dataset)

        losses.append(epoch_loss)

        print(f'Loss: {epoch_loss}')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model, losses
