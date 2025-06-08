# 图像拼图分块聚类

## 项目概述
本项目实现了一个基于CNN的图像拼图分块聚类系统，可以将任意给定的图像划分为网格块，然后使用预训练的CNN模型提取每个块的特征，并通过聚类算法将相似的块归为一类。

## 原理

### 1. 图像分割
将输入图像均匀地划分为8×8或10×10的网格块。

### 2. 特征提取
使用预训练的CNN网络（本项目默认使用ResNet18）从timm库中加载，将其作为特征提取器，提取每个图像块的高级特征表示。

### 3. 相似性计算
- 特征向量标准化
- 使用余弦相似度计算特征向量之间的相似性
- 生成相似性矩阵以可视化块之间的关系

### 4. 自动聚类
本项目提供两种聚类方法：
1. **K-means聚类**：结合轮廓系数（Silhouette Score）自动确定最佳聚类数
2. **亲和力传播聚类（Affinity Propagation）**：自动确定聚类数，不需要预先指定

### 5. 可视化
将聚类结果可视化，在每个图像块上标记聚类标签，并用不同颜色区分不同的聚类。

## 依赖库
- PyTorch 和 torchvision：深度学习框架
- timm：预训练模型库
- scikit-learn：聚类算法
- PIL：图像处理
- matplotlib：可视化
- numpy：数值计算
- tqdm：进度条

## 使用方法

1. 安装依赖：
```bash
pip install torch torchvision timm scikit-learn matplotlib numpy tqdm pillow
```

2. 运行程序：
```bash
python puzzle_blocks_clustering.py --grid-size 8 --method kmeans --images 1.jpeg
```

3. 默认情况下，程序会处理当前目录下的"1.jpeg"、"2.jpeg"和"3.jpeg"三个图像文件，并将结果保存在"results"目录下。

## 重要函数说明

1. **CNNFeatureExtractor**：使用timm库加载预训练CNN模型，用于特征提取
2. **split_image_into_blocks**：将图像划分为网格块
3. **determine_optimal_clusters**：使用轮廓系数确定最佳聚类数
4. **calculate_feature_similarity**：计算特征向量之间的相似性
5. **visualize_clustered_blocks**：可视化聚类结果

## 如何确定最佳聚类数

本项目采用两种方法解决"如何知道将块分为几类是最合适的"：

1. **基于轮廓系数的方法**：轮廓系数是评估聚类质量的指标，值越大表示聚类效果越好。程序会自动尝试2到20个聚类，并选择轮廓系数最高的聚类数。

2. **亲和力传播聚类**：这是一种自动确定聚类数的算法，它通过数据点之间的"消息传递"来确定聚类数和聚类中心。

## 输出结果

对于每个输入图像，程序会生成以下输出：
1. `{image_name}_clustered_kmeans.png`：K-means聚类结果
2. `{image_name}_clustered_affinity.png`：亲和力传播聚类结果
3. `{image_name}_similarity_matrix.png`：图像块特征相似性矩阵热图 
