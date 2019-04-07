import torch
from torch import nn
from torch.utils import data as data_utils
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.preprocessing import MinMaxScaler
from collections import namedtuple
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
import pickle
import os
import shutil

import time

labels = ['ABSZ', 'CPSZ', 'FNSZ', 'GNSZ', 'SPSZ', 'TCSZ', 'TNSZ']
labels_2_num = {labels[i]: i for i in range(len(labels))}
data_tuple = namedtuple('seizure_type_data', ['seizure_type', 'data'])
data_tuple.__qualname__ = 'data_tuple'


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


class Model:
    labels = {
        'ABSZ': 30, 
        'CPSZ': 8, 
        'FNSZ': 3, 
        'GNSZ': 7, 
        'SPSZ': 67, 
        'TCSZ': 60, 
        'TNSZ': 44
    }
    
    def __init__(self, data_dir, model, sklearn_=False, pytorch_=True, is_cuda=False):
        self.dir = data_dir
        self.model = model
        self.is_cuda = is_cuda
        self.have_label = True
        if pytorch_:
            self.pytorch = True
            self.sklearn = False
        else:
            self.pytorch = False
            self.sklearn = True
            
    def read_data(self, files, dir_new, df=False):
        cnt = 0
        if os.path.isdir(dir_new):
            shutil.rmtree(dir_new)
            os.mkdir(dir_new)
        else:
            os.mkdir(dir_new)
            
        if df:
            df_array_flag = True
            lst_y = []
        
        for file in tqdm_notebook(files):
            with open(f'{self.dir}/{file}', 'rb') as f:
                data = pickle.load(f)
                data, label = data[1], data[0]
                if label not in self.labels.keys():
                    continue
                
                np.random.shuffle(data)
                while len(data) < self.labels[label]:
                    data = np.concatenate([data, data])
                
                data = data[:self.labels[label]]
                
                if df and df_array_flag:
                    df_array_flag = False
                    df_array = data
                    lst_y.extend([labels_2_num[label]] * self.labels[label])
                elif df:
                    df_array = np.concatenate([df_array, data])
                    lst_y.extend([labels_2_num[label]] * self.labels[label])
                
                for i in range(len(data)):
                    with open(f'{dir_new}/seiz_{cnt}.pkl', 'wb') as f:
                        pickle.dump(data_tuple(label, data[i]), f)
                        cnt += 1
        if df:
            df_array = np.concatenate([df_array, np.array(lst_y).reshape(-1, 1)], axis=1)
            return pd.DataFrame(df_array, columns=np.arange(901))
        
    def read_data_stds(self, files, dir_new):
        df_array_flag = True
        lst_y = []
        
        for file in tqdm_notebook(files):
            with open(f'{self.dir}/{file}', 'rb') as f:
                data = pickle.load(f)
                data, label = data[1], data[0]
                if label not in self.labels.keys():
                    continue
                
                data = np.array([
                    *list(np.std(data, axis=0)),
                    *list(np.mean(data, axis=0)),
                    *list(np.max(data, axis=0)),
                    *list(np.min(data, axis=0)),
                ]).reshape(1, -1)
                
                if df_array_flag:
                    df_array_flag = False
                    df_array = data
                    lst_y.append(labels_2_num[label])
                else:
                    df_array = np.concatenate([df_array, data], axis=0)
                    lst_y.append(labels_2_num[label])
        
        df_array = np.concatenate([df_array, np.array(lst_y).reshape(-1, 1)], axis=1)
        return pd.DataFrame(df_array, columns=np.arange(df_array.shape[1]))

    def train(self, train):
        print('Starting prepare data...')
        if self.sklearn:
            self.df = self.read_data_stds(files=train, dir_new=f'{self.dir}_')
            # self.df = self.read_data(files=train, dir_new=f'{self.dir}_', df=True)
        else:
            self.dir_new = f'{self.dir}_'
            self.read_data(files=train, dir_new=self.dir_new)
        print('Preparing data finished. Starting train model...')
        
        if self.sklearn:
            self.df.iloc[:, :3600] = MinMaxScaler().fit_transform(self.df.iloc[:, :3600])
            self.model.fit(self.df.iloc[:, :3600], self.df.iloc[:, 3600])
        else:
            dataset_ = Dataset(self.dir_new)
            self.dataloader_ = data_utils.DataLoader(
                dataset=dataset_, 
                batch_size=128,
                shuffle=True,
            )
            if self.is_cuda:
                self.model = self.model.float().cuda()

            loss_fn = nn.CrossEntropyLoss()
            optimizer_ft = optim.Adam(self.model.parameters(), lr=0.001)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=17, gamma=0.1)

            self.model, losses = train_model(self.model, loss_fn, optimizer_ft, self.dataloader_, is_cuda=self.is_cuda, num_epochs=15)
            
        print('Model training finished.')
    
    def predict(self, test, ret_true=False):
        y_true = []
        y_pred = []
        if self.sklearn:
            for file in test:
                with open(f'{self.dir}/{file}', 'rb') as f:
                    data = pickle.load(f)
                y_true.append(labels_2_num[data[0]])
                
                data = data[1]
                np.random.shuffle(data)
                array = np.array(data[:128])
                pred = self.model.predict(MinMaxScaler().fit_transform(array))
                pred = np.array(list(map(int, list(pred))))
                counts = np.bincount(pred)
                
                y_pred.append(np.argmax(counts))
        else:
            for file in test:
                with open(f'{self.dir}/{file}', 'rb') as f:
                    data = pickle.load(f)
                y_true.append(labels_2_num[data[0]])
                
                tensor = torch.Tensor(data[1][:128])
                if self.is_cuda:
                    tensor = tensor.cuda()
                pred = self.model(tensor)
                pred = pred.std(dim=0)
                
                y_pred.append(pred.argmax().cpu().tolist())
                
                
        if ret_true:
            return y_pred, y_true
        else:
            return y_pred
        
    def predict_stds(self, test, ret_true=False):
        if self.sklearn:
            y_true = []
            y_pred = []
            flag = True
            ret_letters = False
            
            for file in test:
                with open(f'{self.dir}/{file}', 'rb') as f:
                    data = pickle.load(f)
                if self.have_label:
                    y_true.append(labels_2_num[data[0]])
                else:
                    ret_true = False
                    ret_letters = True
                data = data[1]
                data = np.array([
                    *list(np.std(data, axis=0)),
                    *list(np.mean(data, axis=0)),
                    *list(np.max(data, axis=0)),
                    *list(np.min(data, axis=0)),
                ]).reshape(1, -1)
                if flag:
                    df_array = data
                    flag = False
                else:
                    df_array = np.concatenate([df_array, data])

            y_pred = self.model.predict(MinMaxScaler().fit_transform(df_array))
        else:
            y_true = []
            y_pred = []
            
            for file in test:
                with open(f'{self.dir}/{file}', 'rb') as f:
                    data = pickle.load(f)
                y_true.append(labels_2_num[data[0]])
                
                tensor = torch.Tensor(data[1][:128])
                if self.is_cuda:
                    tensor = tensor.cuda()
                pred = self.model(tensor)
                pred = pred.std(dim=0)
                
                y_pred.append(pred.argmax().cpu().tolist())

        if ret_true:
            return y_pred, y_true
        elif ret_letters:
            y_pred_cls = []
            for i, y_ in enumerate(y_pred):
                y_pred_cls.append(labels[int(y_)])
            return y_pred_cls
        return y_pred
    
    def predict_test(self, test, have_label=False):
        self.have_label = have_label
        if self.have_label:
            pred, true = self.predict_stds(test, ret_true=True)
            from sklearn.metrics import f1_score
            return f1_score(true, pred, average='weighted')
        else:
            predict = self.predict_stds(test)
            return predict
