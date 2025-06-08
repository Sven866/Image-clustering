import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.functional import crop
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import timm
from tqdm import tqdm
import argparse
from pathlib import Path

class CNNFeatureExtractor:
    def __init__(self, model_name='resnet18', use_gpu=True):
        """
        初始化CNN特征提取器
        
        Args:
            model_name: timm中预训练模型的名称
            use_gpu: 是否使用GPU
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载预训练模型
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    @torch.no_grad()
    def extract_features(self, image):
        """
        提取图像特征
        
        Args:
            image: PIL图像
            
        Returns:
            特征向量
        """
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model(img_tensor)
        return features.squeeze().cpu().numpy()
    
    @torch.no_grad()
    def extract_features_batch(self, images, batch_size=16):
        """
        批量提取图像特征
        
        Args:
            images: PIL图像列表
            batch_size: 批处理大小
            
        Returns:
            特征向量列表
        """
        features_list = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_tensors = torch.stack([self.preprocess(img) for img in batch_images]).to(self.device)
            batch_features = self.model(batch_tensors)
            features_list.append(batch_features.cpu().numpy())
        
        # 拼接所有批次的特征
        features = np.vstack(features_list)
        return features

def split_image_into_blocks(image_path, grid_size=8):
    """
    将图像分割为网格块
    
    Args:
        image_path: 图像路径
        grid_size: 网格大小 (8 表示 8x8 网格, 10 表示 10x10 网格)
        
    Returns:
        blocks: 图像块列表
        block_positions: 每个块的位置 (行, 列)
        original_image: 原始图像
    """
    img = Image.open(image_path)
    width, height = img.size
    
    # 计算每个块的大小
    block_width = width // grid_size
    block_height = height // grid_size
    
    blocks = []
    block_positions = []
    
    for row in range(grid_size):
        for col in range(grid_size):
            # 计算裁剪区域
            left = col * block_width
            top = row * block_height
            right = min(left + block_width, width)  # 确保不超出图像边界
            bottom = min(top + block_height, height)  # 确保不超出图像边界
            
            # 裁剪并添加到块列表
            block = img.crop((left, top, right, bottom))
            blocks.append(block)
            block_positions.append((row, col))
    
    return blocks, block_positions, img

def determine_optimal_clusters(features, max_clusters=20, min_clusters=2):
    """
    使用轮廓系数确定最佳聚类数
    
    Args:
        features: 特征矩阵
        max_clusters: 要测试的最大聚类数
        min_clusters: 要测试的最小聚类数
        
    Returns:
        最佳聚类数
    """
    silhouette_scores = []
    
    # 对特征进行降维以加速计算
    if features.shape[1] > 100:
        pca = PCA(n_components=min(100, features.shape[0]))
        features_reduced = pca.fit_transform(features)
    else:
        features_reduced = features
    
    # 测试不同的聚类数
    for n_clusters in range(min_clusters, min(max_clusters + 1, len(features))):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_reduced)
        
        # 计算轮廓系数
        silhouette_avg = silhouette_score(features_reduced, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"聚类数 {n_clusters}: 轮廓系数 = {silhouette_avg:.3f}")
    
    # 找到轮廓系数最高的聚类数
    if silhouette_scores:
        best_n_clusters = range(min_clusters, min(max_clusters + 1, len(features)))[np.argmax(silhouette_scores)]
        print(f"最佳聚类数: {best_n_clusters}")
        return best_n_clusters
    else:
        print("无法确定最佳聚类数")
        return min(5, len(features))  # 默认返回5个聚类或特征数量

def apply_affinity_propagation(features):
    """
    应用亲和力传播聚类（自动确定聚类数）
    
    Args:
        features: 特征矩阵
        
    Returns:
        聚类标签
    """
    # 使用PCA降维以减少计算复杂度
    pca = PCA(n_components=min(50, features.shape[1], features.shape[0]-1))
    features_pca = pca.fit_transform(features)
    
    # 应用亲和力传播聚类
    af = AffinityPropagation(random_state=42, max_iter=300, convergence_iter=15, damping=0.9)
    cluster_labels = af.fit_predict(features_pca)
    
    n_clusters = len(set(cluster_labels))
    print(f"亲和力传播聚类产生的聚类数: {n_clusters}")
    
    return cluster_labels

def visualize_clustered_blocks(original_image, block_positions, cluster_labels, grid_size=8, output_path=None):
    """
    可视化聚类结果，直接在原始图片的副本上标注编号
    
    Args:
        original_image: 原始图像
        block_positions: 每个块的位置
        cluster_labels: 聚类标签
        grid_size: 网格大小
        output_path: 保存路径
    """
    width, height = original_image.size
    block_width = width // grid_size
    block_height = height // grid_size
    
    # 创建一个新图像用于绘制结果
    result_img = original_image.copy()
    
    # 如果原图不是RGBA模式，转换为RGBA
    if result_img.mode != 'RGBA':
        result_img = result_img.convert('RGBA')
    
    # 创建一个透明图层用于叠加颜色
    overlay = Image.new('RGBA', result_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # 尝试加载字体，如果失败使用默认字体
    try:
        font_size = min(block_width, block_height) // 3
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # 为每个聚类分配一个不同的颜色
    n_clusters = len(set(cluster_labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    colors = [(int(r*255), int(g*255), int(b*255), 128) for r, g, b, _ in colors]  # 转换为RGBA格式，半透明
    
    # 绘制网格线
    for i in range(1, grid_size):
        # 绘制水平线
        draw.line([(0, i * block_height), (width, i * block_height)], fill=(255, 255, 255, 180), width=2)
        # 绘制垂直线
        draw.line([(i * block_width, 0), (i * block_width, height)], fill=(255, 255, 255, 180), width=2)
    
    # 在每个块中绘制标签
    for (row, col), label in zip(block_positions, cluster_labels):
        left = col * block_width
        top = row * block_height
        right = min(left + block_width, width)
        bottom = min(top + block_height, height)
        
        # 绘制半透明的颜色块
        draw.rectangle([left, top, right, bottom], fill=colors[label])
        
        # 计算文本的位置（居中）
        text = str(label)
        # 估计文本宽度，PIL没有直接提供的文本尺寸方法
        text_width = font_size * len(text) // 2
        text_height = font_size
        
        text_x = left + (block_width - text_width) // 2
        text_y = top + (block_height - text_height) // 2
        
        # 绘制文本（白色）
        draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)
    
    # 将半透明层叠加到原图上
    result_img = Image.alpha_composite(result_img, overlay)
    
    # 保存或显示结果
    if output_path:
        # 转换回RGB模式以便保存
        result_img = result_img.convert('RGB')
        result_img.save(output_path)
        print(f"聚类结果保存到: {output_path}")
    
    return result_img

def calculate_feature_similarity(features):
    """
    计算特征之间的相似性矩阵
    
    Args:
        features: 特征矩阵
        
    Returns:
        相似性矩阵
    """
    # 标准化特征
    normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    # 计算余弦相似度
    similarity_matrix = np.dot(normalized_features, normalized_features.T)
    
    return similarity_matrix

def main(image_path, grid_size=8, method='kmeans', output_dir="results", model_name="resnet18", batch_size=16):
    """
    主函数
    
    Args:
        image_path: 图像路径
        grid_size: 网格大小
        method: 聚类方法 ('kmeans', 'affinity')
        output_dir: 输出目录
        model_name: 预训练模型名称
        batch_size: 特征提取批处理大小
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像文件名（不带扩展名）
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 分割图像为网格块
    print(f"将图像分割为 {grid_size}x{grid_size} 网格...")
    blocks, block_positions, original_image = split_image_into_blocks(image_path, grid_size)
    
    # 初始化特征提取器
    print(f"初始化CNN特征提取器 (模型: {model_name})...")
    extractor = CNNFeatureExtractor(model_name=model_name)
    
    # 批量提取特征
    print("提取特征...")
    features = extractor.extract_features_batch(blocks, batch_size=batch_size)
    print(f"特征形状: {features.shape}")
    
    # 计算特征相似性矩阵
    print("计算特征相似性...")
    similarity_matrix = calculate_feature_similarity(features)
    
    # 应用聚类
    print(f"使用 {method} 进行聚类...")
    if method == 'kmeans':
        # 确定最佳聚类数
        n_clusters = determine_optimal_clusters(features)
        
        # 应用K均值聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
    elif method == 'affinity':
        # 使用亲和力传播聚类（自动确定聚类数）
        cluster_labels = apply_affinity_propagation(features)
    else:
        raise ValueError(f"不支持的聚类方法: {method}")
    
    # 可视化聚类结果
    output_path = os.path.join(output_dir, f"{image_name}_clustered_{method}_grid{grid_size}.png")
    result_img = visualize_clustered_blocks(original_image, block_positions, cluster_labels, grid_size, output_path)
    
    # 保存相似性矩阵热图
    plt.figure(figsize=(10, 10))
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('块特征相似性矩阵')
    plt.savefig(os.path.join(output_dir, f"{image_name}_similarity_matrix_grid{grid_size}.png"))
    plt.close()
    
    print("处理完成!")
    return result_img

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='图像拼图分块聚类')
    parser.add_argument('--images', nargs='+', default=["1.jpeg", "2.jpeg", "3.jpeg"],
                        help='要处理的图像文件列表')
    parser.add_argument('--grid-size', type=int, default=8, choices=[8, 10],
                        help='网格大小 (8 或 10)')
    parser.add_argument('--method', choices=['kmeans', 'affinity', 'both'], default='both',
                        help='聚类方法: kmeans, affinity, both')
    parser.add_argument('--output-dir', default='results',
                        help='结果输出目录')
    parser.add_argument('--model', default='resnet18',
                        help='用于特征提取的预训练模型名称')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='特征提取批处理大小')
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 确保输入图像路径有效
    valid_images = []
    for img_path in args.images:
        if os.path.exists(img_path):
            valid_images.append(img_path)
        else:
            print(f"警告: 图像路径 '{img_path}' 不存在，已跳过")
    
    if not valid_images:
        print("错误: 没有有效的图像路径")
        exit(1)
    
    # 处理每个图像
    for image_path in valid_images:
        print(f"\n处理图像: {image_path}")
        
        methods = ['kmeans', 'affinity'] if args.method == 'both' else [args.method]
        
        for method in methods:
            print(f"\n使用 {method} 方法:")
            main(
                image_path, 
                grid_size=args.grid_size, 
                method=method, 
                output_dir=args.output_dir,
                model_name=args.model,
                batch_size=args.batch_size
            ) 