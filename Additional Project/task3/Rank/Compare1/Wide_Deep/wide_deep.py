import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class Linear(nn.Module):
    """
    Linear part
    """
    def __init__(self, input_dim):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1)
    
    def forward(self, x):
        return self.linear(x)

class Dnn(nn.Module):
    """
    Dnn part
    """
    def __init__(self, hidden_units, dropout=0.):
        """
        hidden_units: 列表，每个元素表示每一层的神经单元个数
        dropout: 失活率
        """
        super(Dnn, self).__init__()
        
        self.dnn_network = nn.ModuleList([
            nn.Linear(layer[0], layer[1]) 
            for layer in list(zip(hidden_units[:-1], hidden_units[1:]))
        ])
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        x = self.dropout(x)
        return x

class WideDeep(nn.Module):
    def __init__(self, feature_columns, hidden_units, dnn_dropout=0.):
        super(WideDeep, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_columns
        
        # embedding 
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], 
                                          embedding_dim=feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_cols)
        })
        
        hidden_units.insert(0, len(self.dense_feature_cols) + 
                          len(self.sparse_feature_cols) * 
                          self.sparse_feature_cols[0]['embed_dim'])
        self.dnn_network = Dnn(hidden_units)
        self.linear = Linear(len(self.dense_feature_cols))
        self.final_linear = nn.Linear(hidden_units[-1], 1)
    
    def forward(self, x):
        dense_input = x[:, :len(self.dense_feature_cols)]
        sparse_inputs = x[:, len(self.dense_feature_cols):].long()
        
        sparse_embeds = [
            self.embed_layers['embed_'+str(i)](sparse_inputs[:, i]) 
            for i in range(sparse_inputs.shape[1])
        ]
        sparse_embeds = torch.cat(sparse_embeds, axis=-1)
        
        dnn_input = torch.cat([sparse_embeds, dense_input], axis=-1)
        
        # Wide
        wide_out = self.linear(dense_input)
        
        # Deep
        deep_out = self.dnn_network(dnn_input)
        deep_out = self.final_linear(deep_out)
        
        # out
        outputs = F.sigmoid(0.5 * (wide_out + deep_out))
        return outputs

def plot_metric(dfhistory, metric):
    """绘制训练和验证的metric曲线"""
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.savefig(f'./model/{metric}_history.png')
    plt.close()

def train_model(model, train_loader, val_loader, optimizer, loss_func, 
                metric_func, epochs=10, log_step_freq=10):
    """训练模型的函数"""
    metric_name = 'auc'
    dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, 
                                    "val_loss", "val_"+metric_name])
    print('Start Training...')
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('='*64 + f'{nowtime}')
    
    for epoch in range(1, epochs+1):
        # 训练阶段
        model.train()
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1
        
        train_pred_list = []
        train_label_list = []
        
        for step, (features, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            predictions = model(features)
            loss = loss_func(predictions, labels)
            
            # 收集预测值和真实值
            train_pred_list.extend(predictions.detach().cpu().numpy())
            train_label_list.extend(labels.detach().cpu().numpy())
            
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()
            
            if step % log_step_freq == 0:
                print(f"[step = {step}] loss: {loss_sum/step:.3f}")
        
        # 计算训练集的AUC
        train_auc = roc_auc_score(train_label_list, train_pred_list)
        
        # 验证阶段
        model.eval()
        val_loss_sum = 0.0
        val_pred_list = []
        val_label_list = []
        val_step = 1
        
        for val_step, (features, labels) in enumerate(val_loader, 1):
            with torch.no_grad():
                predictions = model(features)
                val_loss = loss_func(predictions, labels)
                val_pred_list.extend(predictions.cpu().numpy())
                val_label_list.extend(labels.cpu().numpy())
            val_loss_sum += val_loss.item()
        
        # 计算验证集的AUC
        val_auc = roc_auc_score(val_label_list, val_pred_list)
        
        # 记录日志
        info = (epoch, loss_sum/step, train_auc, 
                val_loss_sum/val_step, val_auc)
        dfhistory.loc[epoch-1] = info
        
        print(f"\nEPOCH = {epoch}, loss = {loss_sum/step:.3f}, "
              f"{metric_name} = {train_auc:.3f}, "
              f"val_loss = {val_loss_sum/val_step:.3f}, "
              f"val_{metric_name} = {val_auc:.3f}")
        
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "="*64 + f"{nowtime}")
    
    print('Finished Training...')
    
    # 绘制训练历史
    plot_metric(dfhistory, "loss")
    plot_metric(dfhistory, "auc")
    
    return dfhistory

def prepared_data(file_path):
    """准备数据"""
    # 添加错误检查
    try:
        train = pd.read_csv(file_path + 'train_set.csv')
        val = pd.read_csv(file_path + 'val_set.csv')
        test = pd.read_csv(file_path + 'test_set.csv')
        
        # 打印数据基本信息
        print("训练集形状:", train.shape)
        print("验证集形状:", val.shape)
        print("测试集形状:", test.shape)
        
        # 检查是否有空值
        print("\n是否存在空值:")
        print("训练集:", train.isnull().any().any())
        print("验证集:", val.isnull().any().any())
        print("测试集:", test.isnull().any().any())
        
        trn_x = train.drop(columns='Label').values
        trn_y = train['Label'].values
        val_x = val.drop(columns='Label').values
        val_y = val['Label'].values
        test_x = test.values
        
        fea_col = np.load(file_path + 'fea_col.npy', allow_pickle=True)
        
        return fea_col, (trn_x, trn_y), (val_x, val_y), test_x
        
    except FileNotFoundError as e:
        print(f"错误：找不到数据文件 - {e}")
        raise
    except pd.errors.EmptyDataError:
        print("错误：数据文件为空")
        raise
    except Exception as e:
        print(f"错误：读取数据时发生问题 - {e}")
        raise

def main():
    # 数据准备
    file_path = './preprocessed_data/'
    fea_cols, (trn_x, trn_y), (val_x, val_y), test_x = prepared_data(file_path)
    
    # 构建数据加载器
    dl_train_dataset = TensorDataset(
        torch.tensor(trn_x).float(), 
        torch.tensor(trn_y).float().reshape(-1, 1)
    )
    dl_val_dataset = TensorDataset(
        torch.tensor(val_x).float(), 
        torch.tensor(val_y).float().reshape(-1, 1)
    )

    dl_train = DataLoader(dl_train_dataset, shuffle=True, batch_size=32)
    dl_val = DataLoader(dl_val_dataset, shuffle=True, batch_size=32)

    # 初始化模型
    hidden_units = [256, 128, 64]
    model = WideDeep(fea_cols, hidden_units)
    
    # 打印模型结构
    summary(model, input_size=(32, trn_x.shape[1]))
    
    # 定义损失函数和优化器
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    
    # 训练模型
    history = train_model(model, dl_train, dl_val, optimizer, loss_func, 
                         roc_auc_score, epochs=5)
    
    # 保存模型
    torch.save(model.state_dict(), './model/wide_deep_model.pth')
    
    # 预测测试集
    model.eval()
    with torch.no_grad():
        test_pred = model(torch.tensor(test_x).float())
        y_pred = torch.where(test_pred > 0.5, 
                           torch.ones_like(test_pred), 
                           torch.zeros_like(test_pred))
    
    print("Prediction completed.")

if __name__ == "__main__":
    main() 