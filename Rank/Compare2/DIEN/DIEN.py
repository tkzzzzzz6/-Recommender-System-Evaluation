# -*- coding: utf-8 -*-

from collections import namedtuple
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K
import torch.nn as nn
import torch.nn.functional as F

from contrib.rnn_v2 import dynamic_rnn
from contrib.utils import QAAttGRUCell, VecAttGRUCell

from utils import DenseFeat, SparseFeat, VarLenSparseFeat

tf.compat.v1.disable_eager_execution()   # 这句要加上


# 构建输入层
# 将输入的数据转换成字典的形式，定义输入层的时候让输入层的name和字典中特征的key一致，就可以使得输入的数据和对应的Input层对应
def build_input_layers(feature_columns):
    input_layer_dict = {}

    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_layer_dict[fc.name] = Input(shape=(1,), name=fc.name)
        elif isinstance(fc, DenseFeat):
            input_layer_dict[fc.name] = Input(shape=(fc.dimension, ), name=fc.name)
        elif isinstance(fc, VarLenSparseFeat):
            input_layer_dict[fc.name] = Input(shape=(fc.maxlen, ), name=fc.name)
            
            if fc.length_name:
                input_layer_dict[fc.length_name] = Input((1,), name=fc.length_name, dtype='int32')
    
    return input_layer_dict


# 构建embedding层
def build_embedding_layers(feature_columns, input_layer_dict):
    embedding_layer_dict = {}

    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            embedding_layer_dict[fc.name] = Embedding(fc.vocabulary_size, fc.embedding_dim, name='emb_' + fc.name)
        elif isinstance(fc, VarLenSparseFeat):
            embedding_layer_dict[fc.name] = Embedding(fc.vocabulary_size + 1, fc.embedding_dim, name='emb_' + fc.name, mask_zero=True)

    return embedding_layer_dict

def embedding_lookup(feature_columns, input_layer_dict, embedding_layer_dict):
    embedding_list = []
    
    for fc in feature_columns:
        _input = input_layer_dict[fc]
        _embed = embedding_layer_dict[fc]
        embed = _embed(_input)
        embedding_list.append(embed)

    return embedding_list

# 输入层拼接成列表
def concat_input_list(input_list):
    feature_nums = len(input_list)
    if feature_nums > 1:
        return Concatenate(axis=1)(input_list)
    elif feature_nums == 1:
        return input_list[0]
    else:
        return None

# 将所有的sparse特征embedding拼接
def concat_embedding_list(feature_columns, input_layer_dict, embedding_layer_dict, flatten=False):
    embedding_list = []
    for fc in feature_columns:
        _input = input_layer_dict[fc.name] # 获取输入层 
        _embed = embedding_layer_dict[fc.name] # B x 1 x dim  获取对应的embedding层
        embed = _embed(_input) # B x dim  将input层输入到embedding层中

        # 是否需要flatten, 如果embedding列表最终是直接输入到Dense层中，需要进行Flatten，否则不需要
        if flatten:
            embed = Flatten()(embed)
        
        embedding_list.append(embed)
    
    return embedding_list 

"""Attention NetWork"""
class LocalActivationUnit(Layer):

    def __init__(self, hidden_units=(256, 128, 64), activation='prelu'):
        super(LocalActivationUnit, self).__init__()
        self.hidden_units = hidden_units
        self.linear = Dense(1)
        self.dnn = [Dense(unit, activation=PReLU() if activation == 'prelu' else Dice()) for unit in hidden_units]

    def call(self, inputs):
        # query: B x 1 x emb_dim  keys: B x len x emb_dim
        query, keys = inputs 

        # 获取序列长度
        keys_len, keys_dim = keys.get_shape()[1], keys.get_shape()[2]

        queries = tf.tile(query, multiples=[1, keys_len, 1])   # (None, len * emb_dim)  
        queries = tf.reshape(queries, shape=[-1, keys_len, keys_dim])

        # 将特征进行拼接
        att_input = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1) # B x len x 4*emb_dim

        # 将原始向量与外积结果拼接后输入到一个dnn中
        att_out = att_input
        for fc in self.dnn:
            att_out = fc(att_out) # B x len x att_out

        att_out = self.linear(att_out) # B x len x 1
        att_out = tf.squeeze(att_out, -1) # B x len

        return att_out

class AttentionPoolingLayer(Layer):
    def __init__(self, user_behavior_length, att_hidden_units=(256, 128, 64), return_score=False):
        super(AttentionPoolingLayer, self).__init__()
        self.att_hidden_units = att_hidden_units
        self.local_att = LocalActivationUnit(self.att_hidden_units)
        self.user_behavior_length = user_behavior_length
        self.return_score = return_score

    def call(self, inputs):
        # keys: B x len x emb_dim, queries: B x 1 x emb_dim
        queries, keys = inputs 

        # 获取行为序列embedding的mask矩阵，将Embedding矩阵中的非零元素设置成True，
        key_masks = tf.sequence_mask(self.user_behavior_length, keys.shape[1])  # (None, 1, max_len)  这里注意user_behavior_length是(None,1)
        key_masks = key_masks[:, 0, :]     # 所以上面会多出个1维度来， 这里去掉才行，(None, max_len)

        # 获取行为序列中每个商品对应的注意力权重
        attention_score = self.local_att([queries, keys])   # (None, max_len)

        # 创建一个padding的tensor, 目的是为了标记出行为序列embedding中无效的位置
        paddings = tf.zeros_like(attention_score) # B x len

        # outputs 表示的是padding之后的attention_score
        outputs = tf.where(key_masks, attention_score, paddings) # B x len

        # 将注意力分数与序列对应位置加权求和，这一步可以在
        outputs = tf.expand_dims(outputs, axis=1) # B x 1 x len
        
        if not self.return_score:
            # keys : B x len x emb_dim
            outputs = tf.matmul(outputs, keys) # B x 1 x dim
            outputs = tf.squeeze(outputs, axis=1)

        return outputs

"""兴趣进化网络"""
class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size=None, return_sequence=True):
        super(DynamicGRU, self).__init__()
        self.hidden_size = hidden_size if hidden_size else input_size
        self.return_sequence = return_sequence
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            batch_first=True
        )
    
    def forward(self, inputs):
        # inputs: [sequences, lengths]
        sequences, lengths = inputs
        batch_size = sequences.size(0)
        
        # Pack padded sequence
        packed_sequences = nn.utils.rnn.pack_padded_sequence(
            sequences, 
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Forward pass through GRU
        outputs, _ = self.gru(packed_sequences)
        
        # Unpack sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs,
            batch_first=True
        )
        
        if self.return_sequence:
            return outputs
        else:
            return outputs[:, -1, :]

class AUGRU(nn.Module):
    def __init__(self, input_size, hidden_size=None):
        super(AUGRU, self).__init__()
        self.hidden_size = hidden_size if hidden_size else input_size
        
        # GRU gates
        self.gate_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.gate_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.gate_h = nn.Linear(input_size + hidden_size, hidden_size)
        
    def forward(self, x, hidden, att_score):
        # x: (batch, input_size)
        # hidden: (batch, hidden_size)
        # att_score: (batch, 1)
        
        gates_input = torch.cat([x, hidden], dim=1)
        
        r = torch.sigmoid(self.gate_r(gates_input))
        z = torch.sigmoid(self.gate_z(gates_input))
        
        h_tilde = torch.tanh(self.gate_h(torch.cat([x, r * hidden], dim=1)))
        
        h = (1 - att_score) * hidden + att_score * h_tilde
        
        return h

class DIEN(nn.Module):
    def __init__(self, feature_columns, behavior_feature_list, 
                 behavior_seq_feature_list, use_neg_sample=False, alpha=1.0):
        super(DIEN, self).__init__()
        
        self.feature_columns = feature_columns
        self.behavior_feature_list = behavior_feature_list
        self.behavior_seq_feature_list = behavior_seq_feature_list
        self.use_neg_sample = use_neg_sample
        self.alpha = alpha
        
        # 创建embedding层
        self.embedding_dict = nn.ModuleDict()
        for feat in self.feature_columns:
            self.embedding_dict[feat['name']] = nn.Embedding(
                feat['vocabulary_size'],
                feat['embedding_dim']
            )
        
        # 兴趣抽取层
        self.interest_extractor = DynamicGRU(
            input_size=sum(feat['embedding_dim'] for feat in feature_columns),
            return_sequence=True
        )
        
        # 兴趣进化层
        self.interest_evolution = AUGRU(
            input_size=sum(feat['embedding_dim'] for feat in feature_columns)
        )
        
        # 全连接层
        self.dnn = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 处理输入特征
        sparse_embedding_list = []
        dense_value_list = []
        
        for feat in self.feature_columns:
            if feat['type'] == 'sparse':
                sparse_embedding_list.append(
                    self.embedding_dict[feat['name']](x[feat['name']])
                )
            else:
                dense_value_list.append(x[feat['name']])
        
        # 行为序列处理
        behavior_embedding_list = []
        for feat in self.behavior_seq_feature_list:
            behavior_embedding_list.append(
                self.embedding_dict[feat['name']](x[feat['name']])
            )
        
        # 兴趣抽取
        concat_behavior = torch.cat(behavior_embedding_list, dim=-1)
        rnn_outputs = self.interest_extractor([concat_behavior, x['seq_length']])
        
        # 兴趣进化
        query = torch.cat([
            self.embedding_dict[feat['name']](x[feat['name']])
            for feat in self.behavior_feature_list
        ], dim=-1)
        
        attention_score = torch.matmul(query, rnn_outputs.transpose(-1, -2))
        attention_score = F.softmax(attention_score, dim=-1)
        
        interest = torch.matmul(attention_score, rnn_outputs)
        
        # 最终预测
        concat_feature = torch.cat([
            torch.cat(sparse_embedding_list, dim=-1),
            torch.cat(dense_value_list, dim=-1) if dense_value_list else torch.tensor([]),
            interest
        ], dim=-1)
        
        output = self.dnn(concat_feature)
        
        return output

"""DNN Network"""
class Dice(Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = BatchNormalization(center=False, scale=False)
        
    def build(self, input_shape):
        self.alpha = self.add_weight(shape=(input_shape[-1],), dtype=tf.float32, name='alpha')

    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)
        
        return self.alpha * (1.0-x_p) * x + x_p * x

def get_dnn_logits(dnn_input, hidden_units=(200, 80), activation='prelu'):
    dnns = [Dense(unit, activation=PReLU() if activation == 'prelu' else Dice()) for unit in hidden_units]

    dnn_out = dnn_input
    for dnn in dnns:
        dnn_out = dnn(dnn_out)
    
    # 获取logits
    dnn_logits = Dense(1, activation='sigmoid')(dnn_out)

    return dnn_logits

def auxiliary_loss(h_states, click_seq, noclick_seq, mask):
    """
    计算auxiliary_loss
    :param h_states: 兴趣提取层的隐藏状态的输出h_states  (None, T-1, embed_dim)
    :param click_seq: 下一个时刻用户点击的embedding向量  (None, T-1, embed_dim)
    :param noclick_seq:下一个时刻用户未点击的embedding向量 (None, T-1, embed_dim)
    :param mask: 用户历史行为序列的长度， 注意这里是原seq_length-1，因为最后一个时间步的输出就没法计算了  (None, 1) 
        
    :return:  根据论文的公式，计算出损失，返回回来
    """
    hist_len, _ = click_seq.get_shape().as_list()[1:]    # (T-1, embed_dim) 元组解包的操作， hist_len=T-1
    mask = tf.sequence_mask(mask, hist_len)   # 这是遮盖的操作  (None, 1, T-1)   每一行是bool类型的值， 为FALSE的为填充
    mask = mask[:, 0, :]    # (None, T-1)    
    
    mask = tf.cast(mask, tf.float32)
    
    click_input = tf.concat([h_states, click_seq], -1)    # (None, T-1, 2*embed_dim)
    noclick_input = tf.concat([h_states, noclick_seq], -1)  # (None, T-1, 2*embed_dim)
    
    auxiliary_nn = DNN([100, 50], activation='sigmoid')
    click_prop = auxiliary_nn(click_input)[:, :, 0]            # (None, T-1)
    noclick_prop = auxiliary_nn(noclick_input)[:, :, 0]      # (None, T-1)
    
    click_loss = -tf.reshape(tf.compat.v1.log(click_prop), [-1, tf.shape(click_seq)[1]]) * mask
    noclick_loss = -tf.reshape(tf.compat.v1.log(1.0-noclick_prop), [-1, tf.shape(noclick_seq)[1]]) * mask
    
    aux_loss = tf.reduce_mean(click_loss + noclick_loss)
    
    return aux_loss  

def interest_evolution(concat_behavior, query_input_item, user_behavior_length, 
                      neg_concat_behavior, gru_type="AUGRU", use_neg=True):
    embedding_size = concat_behavior.shape[-1]  # 直接从tensor的shape获取
    
    # 兴趣提取层
    rnn = DynamicGRU(embedding_size, return_sequence=True)
    rnn_outputs = rnn([concat_behavior, user_behavior_length])
    
    aux_loss = None
    use_aux_loss = None
    
    # "AUGRU"并且采用负采样序列方式，这时候要先计算auxiliary_loss
    if gru_type == "AUGRU" and use_neg:
        aux_loss = auxiliary_loss(rnn_outputs[:, :-1, :], 
                                    concat_behavior[:, 1:, :], 
                                    neg_concat_behavior[:, 1:, :],
                                    tf.subtract(user_behavior_length, 1))

    # 兴趣演化层用的GRU， 这时候先得到输出， 然后把Attention的结果直接加权上去
    if gru_type == "GRU":
        rnn_outputs2 = DynamicGRU(embedding_size, return_sequence=True)([rnn_outputs, user_behavior_length])  # (None, max_len, embed_dim)
        hist = AttentionPoolingLayer(user_behavior_length, return_score=False)([query_input_item, rnn_outputs2])
    else:   
        scores = AttentionPoolingLayer(user_behavior_length, return_score=True)([query_input_item, rnn_outputs])
        # 兴趣演化层如果是AIGRU， 把Attention的结果先乘到输入上去，然后再过GRU
        if gru_type == "AIGRU":
            hist = multiply([rnn_outputs, Permute[2, 1](scores)])
            final_state2 = DynamicGRU(embedding_size, gru_type="GRU", return_sequence=False)([hist, user_behavior_length])
        else:  # 兴趣演化层是AUGRU或者AGRU, 这时候， 需要用相应的cell去进行计算了
            final_state2 = DynamicGRU(embedding_size, gru_type=gru_type, return_sequence=False)([rnn_outputs, user_behavior_length, Permute([2, 1])(scores)])
        hist = final_state2
    return hist, aux_loss

"""DIEN NetWork"""
def DIEN(feature_columns, behavior_feature_list, behavior_seq_feature_list, use_neg_sample=False, alpha=1.0):
    
    # 构建输入层
    input_layer_dict = build_input_layers(feature_columns)
    
    # 将Input层转化为列表的形式作为model的输入
    input_layers = list(input_layer_dict.values())       # 各个输入层
    input_keys = list(input_layer_dict.keys())         # 各个列名
    user_behavior_length = input_layer_dict["seq_length"]
    
    # 筛选出特征中的sparse_fea, dense_fea, varlen_fea
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
    
    history_feature_columns = []
    neg_history_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_"+x, behavior_feature_list))
    neg_history_fc_names = list(map(lambda x: "neg_"+x, history_fc_names))
    
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        elif feature_name in neg_history_fc_names:
            neg_history_feature_columns.append(fc)
    
    # 获取dense
    dnn_dense_input = []
    for fc in dense_feature_columns:
        dnn_dense_input.append(input_layer_dict[fc.name])

    # 将所有的dense特征拼接
    dnn_dense_input = concat_input_list(dnn_dense_input)

    # 构建embedding字典
    embedding_layer_dict = build_embedding_layers(feature_columns, input_layer_dict)
    
    # 因为这里最终需要将embedding拼接后直接输入到全连接层(Dense)中, 所以需要Flatten
    dnn_sparse_embed_input = concat_embedding_list(sparse_feature_columns, input_layer_dict, embedding_layer_dict, flatten=True)
    # 将所有sparse特征的embedding进行拼接
    dnn_sparse_input = concat_input_list(dnn_sparse_embed_input)
    
    # 获取当前的行为特征(movie)的embedding，这里有可能有多个行为产生了行为序列，所以需要使用列表将其放在一起
    query_embed_list = embedding_lookup(behavior_feature_list, input_layer_dict, embedding_layer_dict)
    # 获取行为序列(movie_id序列, hist_movie_id) 对应的embedding，这里有可能有多个行为产生了行为序列，所以需要使用列表将其放在一起
    keys_embed_list = embedding_lookup(behavior_seq_feature_list, input_layer_dict, embedding_layer_dict)
    # 把q,k的embedding拼在一块
    query_emb, keys_emb = concat_input_list(query_embed_list), concat_input_list(keys_embed_list)
    
    # 采样的负行为
    neg_uiseq_embed_list = embedding_lookup(neg_history_fc_names, input_layer_dict, embedding_layer_dict)
    neg_concat_behavior = concat_input_list(neg_uiseq_embed_list)
    
    # 兴趣进化层的计算过程
    dnn_seq_input, aux_loss = interest_evolution(keys_emb, query_emb, user_behavior_length, neg_concat_behavior, gru_type="AUGRU")
    
    # 后面的全连接层
    deep_input_embed = Concatenate()([dnn_dense_input, dnn_sparse_input, dnn_seq_input])
    
    # 获取最终dnn的logits
    dnn_logits = get_dnn_logits(deep_input_embed, activation='prelu')
    model = Model(input_layers, dnn_logits)
    
    # 加兴趣提取层的损失  这个比例可调
    if use_neg_sample:
        model.add_loss(alpha * aux_loss)
        
    # 所有变量需要初始化
    tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.global_variables_initializer())
    return model