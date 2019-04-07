import os
import pickle
import numpy as np
from tqdm import tqdm_notebook
from collections import namedtuple
from random import shuffle

data_tuple = namedtuple('seizure_type_data', ['seizure_type', 'data'])
data_tuple.__qualname__ = 'data_tuple'


def preprocess_1():
    """Достаёт сэмплы"""
    cnt = 0

    dir_ = './pp_2_samples'
    if not os.path.isdir(dir_):
        os.mkdir(dir_)

    for i in tqdm_notebook(range(2002)):
        with open(f'./pp_2/seiz_{i}.pkl', 'rb') as f:
            data = pickle.load(f)

        if data[0] == 'MYSZ':
            continue

        for j in range(len(data[1])):
            with open(f'{dir_}/seiz_{cnt}.pkl', 'wb') as f:
                data_ = data_tuple(data[0], np.array([data[1][j]]))
                pickle.dump(data_, f)
                cnt += 1


def preprocess_2():
    """Уменьшает дисбаланс классов"""
    cnt = 0
    dct = {}

    if not os.path.isdir('pp_2_reduced'):
        os.mkdir('pp_2_reduced')
        
    lst = shuffle(list(range(len(os.listdir('pp_2_samples/')))))

    for i in tqdm_notebook(lst):
        with open(f'pp_2_samples/seiz_{i}.pkl', 'rb') as f:
            data = pickle.load(f)

        dct[data[0]] = dct.get(data[0], 0) + 1
        if dct[data[0]] <= 3000:
            with open(f'pp_2_reduced/seiz_{cnt}.pkl', 'wb') as f:
                pickle.dump(data, f)
                cnt += 1
                
             
def preprocess_df():
    pass
