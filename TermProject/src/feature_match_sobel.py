import numpy as np
import os
from typing import Tuple, List
from matplotlib import pyplot as plt
from PIL import Image
from skimage.filters import threshold_otsu

def load_image_data(data_dir: str, max_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """加载带有文件名标签的图像数据
  
    Args:
        data_dir: 数据目录路径
        max_samples: 最大加载样本数（测试用）
      
    Returns:
        tuple: 
        - images: (N, 28, 28) uint8数组
        - labels: (N,) int32数组
    """
    images = []
    labels = []
  
    for filename in os.listdir(data_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
          
        # 从文件名第一个字符获取标签
        label = int(filename[0]) 
        img_path = os.path.join(data_dir, filename)
        img = Image.open(img_path).convert('L') 
        img_array = np.array(img).astype(np.uint8)
      
        if img_array.shape != (28, 28):
            img = img.resize((28, 28))
            img_array = np.array(img).astype(np.uint8)
      
        images.append(img_array)
        labels.append(label)
      
        if max_samples and len(images) >= max_samples:
            break
  
    return np.array(images), np.array(labels)

def load_mnist_data(base_path: str = '../data/MNIST/mnist-part') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载自定义结构的MNIST数据集
  
    Args:
        base_path: 数据集根目录
      
    Returns:
        tuple: 
        - 训练图像 (N, 28, 28)
        - 测试图像 (M, 28, 28)
        - 训练标签 (N,)
        - 测试标签 (M,)
    """
    train_dir = os.path.join(base_path, 'train-part')
    test_dir = os.path.join(base_path, 'test-part')
  
    train_images, train_labels = load_image_data(train_dir)
    test_images, test_labels = load_image_data(test_dir)
  
    print(f"Loaded {len(train_images)} training samples")
    print(f"Loaded {len(test_images)} testing samples")
  
    return train_images, test_images, train_labels, test_labels

def visualize_predictions(test_images, test_labels, predictions, num_samples=12):
    """可视化预测结果"""
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        idx = np.random.randint(0, len(test_images))
        plt.subplot(3, 4, i+1)
        plt.imshow(test_images[idx], cmap='gray')
        plt.title(f"True: {test_labels[idx]}\nPred: {predictions[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('predictions_visualization.png')
    plt.show()

def evaluate_accuracy(
    test_images: np.ndarray,
    test_labels: np.ndarray,
    template_features: np.ndarray,
    template_labels: np.ndarray,
    sample_limit: int = 1000
) -> float:
    """评估模型准确率并记录预测结果"""
    predictions = []
    correct = 0
    limit = min(len(test_images), sample_limit)
  
    for i in range(limit):
        feature = binary_feature_extractor(test_images[i])
        prediction = knn_classifier(feature, template_features, template_labels)
        predictions.append(prediction)
      
        if prediction == test_labels[i]:
            correct += 1
      
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1} samples, current accuracy: {correct/(i+1):.4f}")
  
    visualize_predictions(test_images[:limit], test_labels[:limit], predictions)
  
    return correct / limit

def binary_feature_extractor(image: np.ndarray) -> np.ndarray:
    """改进版特征提取：增加方向感知"""
    # 自适应二值化（使用OTSU算法）
    from skimage.filters import threshold_otsu
    thresh = threshold_otsu(image)
    binary_img = (image > thresh).astype(int)
  
    # 增加边缘特征：使用Sobel算子
    from scipy.ndimage import sobel
    edge_x = sobel(binary_img, axis=0)
    edge_y = sobel(binary_img, axis=1)
    edge_magnitude = np.hypot(edge_x, edge_y)
  
    # 多尺度分块：7x7主网格 + 3x3子网格
    features = []
    for i in range(0, 28, 4):
        for j in range(0, 28, 4):
            # 主区块特征
            block = binary_img[i:i+4, j:j+4]
            active = (block.sum() >= 8).astype(int)
          
            # 边缘密度特征
            edge_block = edge_magnitude[i:i+4, j:j+4]
            edge_feature = (edge_block.sum() > 0).astype(int)
          
            features.extend([active, edge_feature])
  
    return np.array(features)

def build_template_database(
    train_images: np.ndarray, 
    train_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """修正后的模板库构建函数"""
    # 保持原始三维结构 (N, 28, 28)
    feature_vectors = np.array([binary_feature_extractor(img) for img in train_images])
    return feature_vectors, train_labels

def knn_classifier(
    test_feature: np.ndarray,
    template_features: np.ndarray,
    template_labels: np.ndarray,
    k: int = 3  # 增加k值
) -> int:
    """改进版k-NN分类器（加权投票）"""
    # 使用曼哈顿距离
    distances = np.sum(np.abs(template_features - test_feature), axis=1)
  
    # 取前k个最近邻
    k_nearest = np.argpartition(distances, k)[:k]
  
    # 加权投票（距离倒数加权）
    weights = 1 / (distances[k_nearest] + 1e-6)
    votes = np.bincount(template_labels[k_nearest], weights=weights)
  
    return np.argmax(votes)

def main() -> None:
    """主执行流程"""
    X_train, X_test, y_train, y_test = load_mnist_data()
    print("Dataset loaded with shapes:")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    template_db, template_labels = build_template_database(X_train, y_train)
    print(f"Template database built with {len(template_db)} entries")
    
    accuracy = evaluate_accuracy(X_test, y_test, template_db, template_labels)
    print(f"\nFinal test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()