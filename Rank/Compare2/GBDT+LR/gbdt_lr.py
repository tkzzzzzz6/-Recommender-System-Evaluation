import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score, precision_score, recall_score, f1_score
import gc
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

class GBDTLR:
    def __init__(self, num_leaves=31, max_depth=-1, n_estimators=100, learning_rate=0.1):
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.gbdt = None
        self.lr = None
        
    def load_data(self, path='data/'):
        """加载和预处理数据"""
        print("Loading data...")
        df_train = pd.read_csv(path + 'train.csv')
        df_test = pd.read_csv(path + 'test.csv')
        
        # 数据预处理
        df_train.drop(['Id'], axis=1, inplace=True)
        df_test.drop(['Id'], axis=1, inplace=True)
        df_test['Label'] = -1
        data = pd.concat([df_train, df_test])
        data.fillna(-1, inplace=True)
        
        return data
        
    def prepare_data(self, data):
        """准备特征数据"""
        print("Preparing features...")
        continuous_fea = ['I'+str(i+1) for i in range(13)]
        category_fea = ['C'+str(i+1) for i in range(26)]
        
        # 连续特征归一化
        scaler = MinMaxScaler()
        for col in continuous_fea:
            data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
            
        # 类别特征编码
        for col in category_fea:
            onehot_feats = pd.get_dummies(data[col], prefix=col)
            data.drop([col], axis=1, inplace=True)
            data = pd.concat([data, onehot_feats], axis=1)
            
        cols = [col for col in data.columns if col not in ['Label']]
        train = data[data['Label'] != -1]
        target = train['Label']
        test = data[data['Label'] == -1]
        
        return train[cols], test[cols], target
        
    def train_gbdt(self, X_train, y_train, X_val, y_val):
        """训练GBDT模型"""
        print("Training GBDT model...")
        gbdt_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators
        }
        
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
        
        self.gbdt = lgb.train(
            gbdt_params,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            callbacks=[lgb.early_stopping(100)]
        )
        
    def train_lr(self, X_train, y_train, X_val, y_val):
        """训练LR模型"""
        print("Training LR model...")
        self.lr = LogisticRegression(penalty='l2', C=1.0)
        self.lr.fit(X_train, y_train)
        
    def transform_features(self, data):
        """使用GBDT转换特征"""
        print("Transforming features using GBDT...")
        gbdt_feats = self.gbdt.predict(data, pred_leaf=True)
        gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats.shape[1])]
        df_gbdt_feats = pd.DataFrame(gbdt_feats, columns=gbdt_feats_name)
        return df_gbdt_feats
        
    def evaluate(self, X, y_true):
        """评估模型性能"""
        print("\nEvaluating model performance...")
        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # 计算各种评估指标
        auc = roc_auc_score(y_true, y_pred_proba)
        logloss = log_loss(y_true, y_pred_proba)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"AUC: {auc:.4f}")
        print(f"Log Loss: {logloss:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return {
            'auc': auc,
            'logloss': logloss,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def predict_proba(self, X):
        """预测概率"""
        gbdt_feats = self.transform_features(X)
        return self.lr.predict_proba(gbdt_feats)[:, 1]
    
    def fit(self, X, y):
        """训练模型"""
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 训练GBDT
        self.train_gbdt(X_train, y_train, X_val, y_val)
        
        # 转换特征
        X_train_gbdt = self.transform_features(X_train)
        X_val_gbdt = self.transform_features(X_val)
        
        # 训练LR
        self.train_lr(X_train_gbdt, y_train, X_val_gbdt, y_val)
        
        # 评估模型
        self.evaluate(X_val, y_val)
        
        return self

def main():
    # 创建模型实例
    model = GBDTLR()
    
    # 加载数据
    data = model.load_data()
    
    # 准备特征
    train_data, test_data, target = model.prepare_data(data)
    
    # 训练模型
    model.fit(train_data, target)
    
    # 预测测试集
    test_pred = model.predict_proba(test_data)
    
    # 保存预测结果
    submission = pd.DataFrame({
        'Id': range(len(test_pred)),
        'Predicted': test_pred
    })
    submission.to_csv('submission.csv', index=False)
    print("\nPrediction saved to submission.csv")

if __name__ == "__main__":
    main() 