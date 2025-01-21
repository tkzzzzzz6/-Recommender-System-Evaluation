import datetime
import numpy as np
import pandas as pd

import torch 
from torch.utils.data import TensorDataset, Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchinfo import summary
# from torchkeras import summary, Model

from sklearn.metrics import auc, roc_auc_score, roc_curve

import warnings
warnings.filterwarnings('ignore')

# 导入数据， 数据已经处理好了 preprocess/下
train_set = pd.read_csv('preprocessed_data/train_set.csv')
val_set = pd.read_csv('preprocessed_data/val_set.csv')
test_set = pd.read_csv('preprocessed_data/test.csv')

val_set.head()

# 这里需要把特征分成数值型和离散型， 因为后面的模型里面离散型的特征需要embedding， 而数值型的特征直接进入了stacking层， 处理方式会不一样
data_df = pd.concat((train_set, val_set, test_set))

dense_feas = ['I'+str(i) for i in range(1, 14)]
sparse_feas = ['C'+str(i) for i in range(1, 27)]

# 定义一个稀疏特征的embedding映射， 字典{key: value}, key表示每个稀疏特征， value表示数据集data_df对应列的不同取值个数， 作为embedding输入维度
sparse_feas_map = {}
for key in sparse_feas:
    sparse_feas_map[key] = data_df[key].nunique()

feature_info = [dense_feas, sparse_feas, sparse_feas_map]  # 这里把特征信息进行封装， 建立模型的时候作为参数传入

train_set.columns
test_set.columns

# 修改数据加载部分的代码
# 把数据构建成数据管道
dl_train_dataset = TensorDataset(
    torch.tensor(train_set.drop(columns='Label').values).float(), 
    torch.tensor(train_set['Label'].values).float().reshape(-1, 1)  # 修改这里，确保标签是2D的
)
dl_val_dataset = TensorDataset(
    torch.tensor(val_set.drop(columns='Label').values).float(), 
    torch.tensor(val_set['Label'].values).float().reshape(-1, 1)    # 修改这里，确保标签是2D的
)

dl_train = DataLoader(dl_train_dataset, shuffle=True, batch_size=16)
dl_valid = DataLoader(dl_val_dataset, shuffle=True, batch_size=16)


# 自定义一个残差块
class Residual_block(nn.Module):
    """
    Define Residual_block
    """
    def __init__(self, hidden_unit, dim_stack):
        super(Residual_block, self).__init__()
        self.linear1 = nn.Linear(dim_stack, hidden_unit)
        self.linear2 = nn.Linear(hidden_unit, dim_stack)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        orig_x = x.clone()
        x = self.linear1(x)
        x = self.linear2(x)
        outputs = self.relu(x + orig_x)
        return outputs