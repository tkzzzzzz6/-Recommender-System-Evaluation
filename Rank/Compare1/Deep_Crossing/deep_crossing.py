import numpy as np
import pandas as pd
import torch 
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.filterwarnings('ignore')

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        orig_x = x.clone()
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        return x + orig_x

class DeepCrossing(nn.Module):
    def __init__(self, feature_info, embedding_dim=8, hidden_units=None, dropout=0.):
        """
        DeepCrossing模型
        :param feature_info: [dense_features, sparse_features, sparse_feature_map]
        :param embedding_dim: embedding维度
        :param hidden_units: 隐藏层单元数，列表形式
        """
        super(DeepCrossing, self).__init__()
        self.dense_features, self.sparse_features, self.sparse_feature_map = feature_info
        
        # Embedding层
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(key): nn.Embedding(num_embeddings=val, 
                                            embedding_dim=embedding_dim)
            for key, val in self.sparse_feature_map.items()
        })
        
        # 计算拼接后的维度
        embed_dim = len(self.sparse_features) * embedding_dim
        dense_dim = len(self.dense_features)
        total_dim = embed_dim + dense_dim
        
        # 残差网络层
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(total_dim) for _ in range(3)
        ])
        
        # 全连接层
        if hidden_units is None:
            hidden_units = [128, 64, 32]
        self.dnn = nn.ModuleList([
            nn.Linear(total_dim if i == 0 else hidden_units[i-1], unit) 
            for i, unit in enumerate(hidden_units)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_units[-1], 1)
        
    def forward(self, x):
        # 分离连续特征和离散特征
        dense_inputs = x[:, :len(self.dense_features)]
        sparse_inputs = x[:, len(self.dense_features):].long()
        
        # Embedding
        sparse_embeds = [
            self.embed_layers['embed_'+str(feat)](sparse_inputs[:, i])
            for i, feat in enumerate(self.sparse_features)
        ]
        sparse_embeds = torch.cat(sparse_embeds, dim=-1)
        
        # 特征拼接
        x = torch.cat([sparse_embeds, dense_inputs], dim=-1)
        
        # 残差网络
        for block in self.residual_blocks:
            x = block(x)
        
        # DNN层
        for linear in self.dnn:
            x = linear(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # 输出层
        outputs = torch.sigmoid(self.final_linear(x))
        return outputs

def plot_metric(dfhistory, metric):
    """绘制训练和验证的metric曲线"""
    plt.figure(figsize=(10, 6))
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

def train_model(model, train_loader, val_loader, optimizer, loss_func, epochs=10):
    """训练模型"""
    metric_name = 'auc'
    dfhistory = pd.DataFrame(columns=['epoch', 'loss', metric_name, 
                                    'val_loss', 'val_'+metric_name])
    print('Start Training...')
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('='*64 + f'{nowtime}')
    
    for epoch in range(1, epochs+1):
        # 训练阶段
        model.train()
        total_loss = 0
        train_predictions = []
        train_labels = []
        
        for features, labels in train_loader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = loss_func(predictions, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_predictions.extend(predictions.detach().numpy())
            train_labels.extend(labels.numpy())
        
        # 计算训练集指标
        train_auc = roc_auc_score(train_labels, train_predictions)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                predictions = model(features)
                loss = loss_func(predictions, labels)
                
                val_loss += loss.item()
                val_predictions.extend(predictions.numpy())
                val_labels.extend(labels.numpy())
        
        # 计算验证集指标
        val_auc = roc_auc_score(val_labels, val_predictions)
        
        # 记录日志
        info = (epoch, total_loss/len(train_loader), train_auc, 
                val_loss/len(val_loader), val_auc)
        dfhistory.loc[epoch-1] = info
        
        print(f"\nEPOCH = {epoch}")
        print(f"Loss = {total_loss/len(train_loader):.3f}")
        print(f"{metric_name} = {train_auc:.3f}")
        print(f"val_loss = {val_loss/len(val_loader):.3f}")
        print(f"val_{metric_name} = {val_auc:.3f}")
        
    print('Finished Training...')
    
    # 绘制训练历史
    plot_metric(dfhistory, "loss")
    plot_metric(dfhistory, "auc")
    
    return dfhistory

def prepare_data(file_path='./preprocessed_data/'):
    """准备数据"""
    # 读取数据
    train_set = pd.read_csv(file_path + 'train_set.csv')
    val_set = pd.read_csv(file_path + 'val_set.csv')
    test_set = pd.read_csv(file_path + 'test.csv')
    
    # 分离特征
    dense_features = ['I'+str(i) for i in range(1, 14)]
    sparse_features = ['C'+str(i) for i in range(1, 27)]
    
    # 构建特征映射
    data_df = pd.concat([train_set, val_set, test_set])
    sparse_feature_map = {
        feat: data_df[feat].nunique() 
        for feat in sparse_features
    }
    
    # 准备数据集
    X_train = train_set.drop('Label', axis=1).values
    y_train = train_set['Label'].values
    X_val = val_set.drop('Label', axis=1).values
    y_val = val_set['Label'].values
    X_test = test_set.values
    
    return [dense_features, sparse_features, sparse_feature_map], \
           (X_train, y_train), (X_val, y_val), X_test

def main():
    # 准备数据
    feature_info, (X_train, y_train), (X_val, y_val), X_test = prepare_data()
    
    # 构建数据加载器
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).reshape(-1, 1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val).reshape(-1, 1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    
    # 初始化模型
    model = DeepCrossing(feature_info)
    
    # 打印模型结构
    summary(model, input_size=(256, X_train.shape[1]))
    
    # 定义损失函数和优化器
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    history = train_model(model, train_loader, val_loader, optimizer, loss_func)
    
    # 保存模型
    torch.save(model.state_dict(), './model/deep_crossing_model.pth')
    
    # 预测测试集
    model.eval()
    with torch.no_grad():
        test_pred = model(torch.FloatTensor(X_test))
    
    print("Prediction completed.")

if __name__ == "__main__":
    main() 