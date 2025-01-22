# 推荐系统实践项目

本项目是推荐系统的实践项目，包含了召回和排序两个主要模块的多种算法实现。

### 背景
在这个推荐系统项目中，多个排序模型（DeepFM、Deep Crossing、GBDT+LR、Wide&Deep、DCN、DIN）已经实现并训练。数据量的多少对模型的性能有着显著的影响。为了在单机环境下进行模型训练和验证,模型使用了规模较小的数据集,这可能导致推荐系统面临冷启动问题,即由于用户行为数据不足而难以生成准确的推荐结果。

### 任务

1. 选择和运行排序模型
- 从项目中选择一个排序模型(DeepFM、Deep Crossing、GBDT+LR、Wide&Deep、DCN或DIN)
- 按照以下步骤配置环境并运行:
  - 创建Python 3.8虚拟环境
  - 安装模型所需的依赖包(PyTorch/TensorFlow等)
  - 运行模型训练脚本
- 记录和保存实验结果:
  - 模型训练过程中的loss曲线
  - 评估指标(AUC、Loss等)
  - 模型文件和预测结果
  - 实验配置和参数设置

2. Rank文件夹中分别有Compare1(有4个模型,选择两个模型即可)与Compare2文件夹，分别含有两种排序模型,按照任务一的步骤配置环境,运行并比较所选模型的性能,并分析在数据不足的情况下各模型为什么有不同差异的表现,哪个模型适合应用在数据不足的情况下。并思考如何在数据不足的情况下的解决方案(不要求改进模型和代码,用文字阐述即可,重点在于思考的过程)。

3. 协同过滤(Collaborative Filtering)是推荐系统中常用的推荐算法之一

#### 1. 理解现有模型
- 为每个排序模型提供一个简要概述
- 包括其关键特点和在推荐系统中的适用场景

#### 2. 分析数据有限的影响
- 使用项目中的代码，选择
- 记录并比较它们的性能指标（如准确率、召回率）
- 分析数据量有限对模型性能的具体影响

#### 3. 提出优化策略
基于分析，提出策略来优化模型在数据有限情况下的性能：

##### 模型层面
- 调整模型超参数，如学习率或正则化强度
- 采用模型融合，结合多个模型的预测结果
- 采用元学习方法，提高模型在小数据集上的泛化能力
- 使用多任务学习提升模型泛化性

##### 数据层面  
- 增加特征工程，利用更多用户或物品的元数据
- 使用数据增强技术，如过采样、欠采样等
- 使用自监督学习方法，充分利用未标记数据
- 引入知识图谱增强特征表示
- 结合多模态信息(图像、文本等)丰富特征
- 使用对比学习提升表示学习效果
- 加入因果推断减少数据偏差

#### 4. 实施与验证
- 在项目代码的基础上，实施提出的优化策略
- 在相同的小数据集上重新运行优化后的模型
- 记录并比较优化前后的性能指标

#### 5. 结果讨论与总结
- 分析实验结果，讨论各优化策略的有效性
- 总结发现，并提出未来工作的方向或进一步优化的建议

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
