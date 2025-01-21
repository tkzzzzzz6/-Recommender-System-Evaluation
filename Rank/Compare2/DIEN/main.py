import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from DIEN import DIEN
from utils import SparseFeat, DenseFeat, VarLenSparseFeat
from random import sample
import warnings
warnings.filterwarnings('ignore')

def get_neg_click(data_df, neg_num=10):
    """获取负样本"""
    try:
        movies_np = data_df['hist_movie_id'].values
        
        movie_list = []
        for movies in movies_np:
            if isinstance(movies, str):  # 确保输入是字符串
                movie_list.extend([x for x in movies.split(',') if x != '0'])

        movies_set = set(movie_list) 

        neg_movies_list = []
        for movies in movies_np:
            if isinstance(movies, str):
                hist_movies = set([x for x in movies.split(',') if x != '0'])
                neg_movies_set = movies_set - hist_movies
                neg_num = min(neg_num, len(neg_movies_set))  # 确保不超过可用数量
                if neg_num > 0:
                    neg_movies = sample(list(neg_movies_set), neg_num)  # 将集合转换为列表
                    neg_movies_list.append(','.join(map(str, neg_movies)))
                else:
                    neg_movies_list.append('0')  # 如果没有可用的负样本，使用0
            else:
                neg_movies_list.append('0')

        return pd.Series(neg_movies_list)
    except Exception as e:
        print(f"Error in get_neg_click: {e}")
        print(f"Data sample: {data_df['hist_movie_id'].head()}")
        raise

def prepare_data():
    """准备数据"""
    # 读取数据
    samples_data = pd.read_csv("data/movie_sample.txt", sep="\t", header=None)
    samples_data.columns = ["user_id", "gender", "age", "hist_movie_id", 
                          "hist_len", "movie_id", "movie_type_id", "label"]

    # 构建特征列
    feature_columns = [
        SparseFeat('user_id', max(samples_data["user_id"])+1, embedding_dim=8),
        SparseFeat('gender', max(samples_data["gender"])+1, embedding_dim=8),
        SparseFeat('age', max(samples_data["age"])+1, embedding_dim=8),
        SparseFeat('movie_id', max(samples_data["movie_id"])+1, embedding_dim=8),
        SparseFeat('movie_type_id', max(samples_data["movie_type_id"])+1, embedding_dim=8),
        DenseFeat('hist_len', 1)
    ]

    feature_columns += [
        VarLenSparseFeat(
            SparseFeat('hist_movie_id', 
                      vocabulary_size=max(samples_data["movie_id"])+1,
                      embedding_dim=8),
            maxlen=50,
            length_name='seq_length'
        )
    ]

    # 数据处理
    X = samples_data[["user_id", "gender", "age", "hist_movie_id", 
                     "hist_len", "movie_id", "movie_type_id"]]
    y = samples_data["label"]

    # 负采样
    X['neg_hist_movie_id'] = get_neg_click(X, neg_num=50)
    behavior_len = np.array([len([int(i) for i in l.split(',') if int(i) != 0]) 
                           for l in X['hist_movie_id']])

    # 构建输入数据
    X_dict = {
        "user_id": X["user_id"].values,
        "gender": X["gender"].values,
        "age": X["age"].values,
        "hist_movie_id": np.array([[int(i) for i in l.split(',')] 
                                     for l in X["hist_movie_id"]]),
        "neg_hist_movie_id": np.array([[int(i) for i in l.split(',')] 
                                         for l in X["neg_hist_movie_id"]]),
        "seq_length": behavior_len,
        "hist_len": X["hist_len"].values,
        "movie_id": X["movie_id"].values,
        "movie_type_id": X["movie_type_id"].values
    }

    return X_dict, y.values, feature_columns

def train_model(model, X_dict, y, batch_size=64, epochs=10):
    """训练模型"""
    history = model.fit(X_dict, y, batch_size=batch_size, epochs=epochs,
                       validation_split=0.2, callbacks=[
                           ModelCheckpoint('dien_model.h5', save_best_only=True)
                       ])
    return history.history

def plot_metrics(history):
    """绘制训练指标"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    
    plt.subplot(122)
    plt.plot(history['auc'])
    plt.plot(history['val_auc'])
    plt.title('Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def main():
    # 准备数据
    X_dict, y, feature_columns = prepare_data()
    
    # 定义模型
    behavior_feature_list = ['movie_id']
    behavior_seq_feature_list = ['hist_movie_id']
    
    model = DIEN(feature_columns, 
                 behavior_feature_list,
                 behavior_seq_feature_list,
                 use_neg_sample=True)
    
    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['AUC'])
    
    # 训练模型
    history = train_model(model, X_dict, y, batch_size=64, epochs=10)
    
    # 绘制训练指标
    plot_metrics(history)

if __name__ == "__main__":
    main()