import torch
from torch.utils import data
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import pickle
import os

import time

use_gpu = False
labels = ['ABSZ', 'CPSZ', 'FNSZ', 'GNSZ', 'SPSZ', 'TCSZ', 'TNSZ']
labels_2_num = {labels[i]: i for i in range(len(labels))}


class Dataset(data.Dataset):
    def __init__(self, path):
        self.path = path

    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, idx):
        with open(f'{self.path}/seiz_{idx}.pkl', 'rb') as f:
            data = pickle.load(f)
            return {'data': data[1], 'label': labels_2_num[data[0]]}

dataset_ = Dataset('pp_2_reduced')


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

def train_model(model, criterion, optimizer, scheduler, dataloader_, num_epochs=25):
    since = time.time()
    
    losses = []
    model.train(True)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0
       # scheduler.step()

        for data in dataloader_:
            inputs, labels = data['data'].float(), data['label']
            inputs = inputs.view(inputs.size(0), -1)

            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            
            loss = criterion(model(inputs), labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataset_)

        losses.append(epoch_loss)

        print(f'Loss: {epoch_loss}')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model, losses
