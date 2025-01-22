# 推荐系统实践项目

本项目是推荐系统的实践项目，包含了召回和排序两个主要模块的多种算法实现。


## 项目结构

```
.
├── Rank/ # 排序模块
│ ├── Compare1/ # 第一组排序模型比较
│ │ ├── DeepFM/ # DeepFM模型
│ │ ├── Deep_Crossing/ # Deep Crossing模型
│ │ ├── GBDT+LR/ # GBDT+LR模型
│ │ └── Wide_Deep/ # Wide&Deep模型
│ └── Compare2/ # 第二组排序模型比较
│ ├── DCN/ # Deep&Cross Network模型
│ └── DIN/ # Deep Interest Network模型
├── Recall/ # 召回模块
│ ├── CollaborativeFiltering/ # 协同过滤
│ └── ContentBasedRecommend/ # 基于内容的推荐
└── paper/ # 相关论文和文档
```

## 环境要求

- Python >= 3.6
- PyTorch >= 1.7.0 (用于深度学习模型)
- TensorFlow >= 2.4.0 (用于部分深度学习模型)
- scikit-learn >= 0.24.2
- LightGBM >= 3.3.5 (用于GBDT+LR模型)
- 其他依赖见各模型目录下的requirements.txt

## 安装说明

1. 克隆项目

```bash
git clone https://github.com/tkzzzzzz6/Recommender-System-Evaluation
cd Recommender-System-Evaluation
```

2. 创建虚拟环境（推荐）
```bash
conda create -n [虚拟环境名称] python=3.8
conda activate [虚拟环境名称]
```

3. 安装依赖
```bash
# 安装特定模型的依赖
cd Rank/Compare1/DeepFM
pip install -r requirements.txt
```

## 使用说明

### 排序模块

#### Compare1

1. DeepFM模型
```bash
cd Rank/Compare1/DeepFM
python deepfm.py
```

2. Wide&Deep模型
```bash
cd Rank/Compare1/Wide_Deep
python wide_deep.py
```

3. Deep_Crossing模型
```bash
cd Rank/Compare1/Deep_Crossing
python deep_crossing.py
```

4. GBDT+LR模型
```bash
cd Rank/Compare1/GBDT+LR
python gbdt_lr.py
```

#### Compare2

1. DCN模型
```bash
cd Rank/Compare1/DCN
python DCN.py
```

2. DIN模型
```bash
cd Rank/Compare1/DIN
python DIN.py
```

### 召回模块

1. 协同过滤
```bash
cd Recall/CollaborativeFiltering
python collaborative_filtering.py
```

2. 基于内容的推荐
```bash
cd Recall/ContentBasedRecommend
python ContentBased_Recommend.py
```


## 数据说明

- 每个模型目录下的 `data/` 文件夹需要包含相应的训练数据
- 数据格式和要求见各模型目录下的说明文档

## 模型评估

各模型都包含了对应的评估指标：
- AUC
- Log Loss

评估结果会保存在各模型的 `model/` 目录下。

## 注意事项

1. 确保数据文件位于正确的位置
2. 创建必要的输出目录（如model/）
4. 建议使用虚拟环境避免依赖冲突


## 参考资料

相关论文和参考资料见 paper/ 目录
