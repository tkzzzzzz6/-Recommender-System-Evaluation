import datetime
import numpy as np
import pandas as pd
import torch 
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from sklearn.metrics import auc, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# 首先，自定义一个残差块
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

# 定义deep Crossing 网络
class DeepCrossing(nn.Module):
    def __init__(self, feature_info, hidden_units, dropout=0., embed_dim=10, output_dim=1):
        """
        DeepCrossing：
            feature_info: 特征信息（数值特征，类别特征，类别特征embedding映射)
            hidden_units: 列表，隐藏单元的个数(多层残差那里的)
            dropout: Dropout层的失活比例
            embed_dim: embedding维度
        """
        super(DeepCrossing, self).__init__()
        self.dense_feas, self.sparse_feas, self.sparse_feas_map = feature_info
        
        # embedding层
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=embed_dim)
            for key, val in self.sparse_feas_map.items()
        })
        
        # 统计embedding_dim的总维度
        embed_dim_sum = sum([embed_dim]*len(self.sparse_feas))
        
        # stack layers的总维度
        dim_stack = len(self.dense_feas) + embed_dim_sum
        
        # 残差层
        self.res_layers = nn.ModuleList([
            Residual_block(unit, dim_stack) for unit in hidden_units
        ])
        
        # dropout层
        self.res_dropout = nn.Dropout(dropout)
        
        # 线性层
        self.linear = nn.Linear(dim_stack, output_dim)
    
    def forward(self, x):
        dense_inputs, sparse_inputs = x[:, :13], x[:, 13:]
        sparse_inputs = sparse_inputs.long()
        sparse_embeds = [self.embed_layers['embed_'+key](sparse_inputs[:, i]) 
                        for key, i in zip(self.sparse_feas_map.keys(), range(sparse_inputs.shape[1]))]   
        sparse_embed = torch.cat(sparse_embeds, axis=-1)
        stack = torch.cat([sparse_embed, dense_inputs], axis=-1)
        r = stack
        for res in self.res_layers:
            r = res(r)
        
        r = self.res_dropout(r)
        outputs = F.sigmoid(self.linear(r))
        return outputs

def train_model(net, dl_train, dl_valid, optimizer, loss_func, metric_func, epochs=4, log_step_freq=10):
    """
    训练模型的函数
    """
    metric_name = 'auc'
    dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_"+metric_name])
    print('Start Training...')
    nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('========='*8 + "%s" %nowtime)

    for epoch in range(1, epochs+1):
        # 训练阶段
        net.train()
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1
        
        for step, (features, labels) in enumerate(dl_train, 1):
            optimizer.zero_grad()
            predictions = net(features)
            loss = loss_func(predictions, labels)
            try:
                metric = metric_func(predictions, labels)
            except ValueError:
                continue
            
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()
            metric_sum += metric
            if step % log_step_freq == 0:
                print(("[step = %d] loss: %.3f, "+metric_name+": %.3f") %
                      (step, loss_sum/step, metric_sum/step))
        
        # 验证阶段
        net.eval()
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1
        
        for val_step, (features, labels) in enumerate(dl_valid, 1):
            with torch.no_grad():
                predictions = net(features)
                val_loss = loss_func(predictions, labels)
                try:
                    val_metric = metric_func(predictions, labels)
                except ValueError:
                    continue
            val_loss_sum += val_loss.item()
            val_metric_sum += val_metric
        
        # 记录日志
        info = (epoch, loss_sum/step, metric_sum/step, 
                val_loss_sum/val_step, val_metric_sum/val_step)
        dfhistory.loc[epoch-1] = info
        
        print(("\nEPOCH = %d, loss = %.3f,"+ metric_name + 
              "  = %.3f, val_loss = %.3f, "+"val_"+ metric_name+" = %.3f") 
              %info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"==========="*8 + "%s"%nowtime)
            
    print('Finished Training...')
    return dfhistory

def main():
    # 加载数据
    train_set = pd.read_csv('preprocessed_data/train_set.csv')
    val_set = pd.read_csv('preprocessed_data/val_set.csv')
    test_set = pd.read_csv('preprocessed_data/test.csv')

    # 特征处理
    data_df = pd.concat((train_set, val_set, test_set))
    dense_feas = ['I'+str(i) for i in range(1, 14)]
    sparse_feas = ['C'+str(i) for i in range(1, 27)]
    
    sparse_feas_map = {}
    for key in sparse_feas:
        sparse_feas_map[key] = data_df[key].nunique()
    
    feature_info = [dense_feas, sparse_feas, sparse_feas_map]

    # 准备数据
    dl_train_dataset = TensorDataset(
        torch.tensor(train_set.drop(columns='Label').values).float(), 
        torch.tensor(train_set['Label'].values).float().reshape(-1, 1)
    )
    dl_val_dataset = TensorDataset(
        torch.tensor(val_set.drop(columns='Label').values).float(), 
        torch.tensor(val_set['Label'].values).float().reshape(-1, 1)
    )

    dl_train = DataLoader(dl_train_dataset, shuffle=True, batch_size=16)
    dl_valid = DataLoader(dl_val_dataset, shuffle=True, batch_size=16)

    # 初始化模型
    hidden_units = [256, 128, 64, 32]
    net = DeepCrossing(feature_info, hidden_units)
    
    # 定义损失函数和优化器
    def auc(y_pred, y_true):
        pred = y_pred.data.cpu().numpy()
        y = y_true.data.cpu().numpy()
        return roc_auc_score(y, pred)

    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
    
    # 训练模型
    history = train_model(net, dl_train, dl_valid, optimizer, loss_func, auc)
    
    # 保存模型
    torch.save(net.state_dict(), './model/deep_crossing_model.pth')

if __name__ == "__main__":
    main() 