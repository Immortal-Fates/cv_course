import numpy as np
import os
from typing import Tuple, List
from matplotlib import pyplot as plt
from PIL import Image

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
  
    # 可视化预测结果
    visualize_predictions(test_images[:limit], test_labels[:limit], predictions)
  
    return correct / limit

def binary_feature_extractor(image: np.ndarray) -> np.ndarray:
    """实现7x7网格二值特征提取。
  
    Args:
        image: 输入图像数组 (28, 28)，取值范围0-255
      
    Returns:
        49维二值特征向量，元素为0或1
    """
    binary_img = (image > 128).astype(int)
    blocks = binary_img.reshape(7, 4, 7, 4)
    activated_blocks = (blocks.sum(axis=(1, 3)) > 8).astype(int)
  
    return activated_blocks.flatten()

def build_template_database(
    train_images: np.ndarray, 
    train_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """构建特征模板数据库。
  
    Args:
        train_images: 训练图像数组 (N, 28, 28)
        train_labels: 对应标签数组 (N,)
      
    Returns:
        tuple:
        - 特征矩阵 (N, 49)
        - 标签数组 (N,)
    """
    feature_vectors = np.apply_along_axis(
        binary_feature_extractor, 
        1, 
        train_images.reshape(-1, 28*28)
    )
    return feature_vectors, train_labels

def knn_classifier(
    test_feature: np.ndarray,
    template_features: np.ndarray,
    template_labels: np.ndarray
) -> int:
    """k-NN分类器实现（k=1）。
  
    Args:
        test_feature: 测试样本特征向量 (49,)
        template_features: 模板特征矩阵 (M, 49)
        template_labels: 模板标签数组 (M,)
      
    Returns:
        预测的数字类别
    """
    # 计算欧氏距离
    distances = np.linalg.norm(template_features - test_feature, axis=1)
  
    # 找到最近邻索引
    nearest_idx = np.argmin(distances)
  
    return template_labels[nearest_idx]

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