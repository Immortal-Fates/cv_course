import numpy as np
import os
from typing import Tuple, List
from matplotlib import pyplot as plt
from PIL import Image
from skimage.filters import threshold_otsu
import cv2
from sklearn.cluster import MiniBatchKMeans


class SIFTFeatureExtractor:
    def __init__(self, vocab_size=50):
        self.sift = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=5)
        self.vocab_size = vocab_size
        self.kmeans = MiniBatchKMeans(
            n_clusters=vocab_size,
            batch_size=3584,  # 修改点1：增大batch_size
            n_init=3  # 修改点2：显式设置n_init
        )
        self.is_vocab_built = False

    def build_vocabulary(self, images):
        """构建视觉词袋"""
        descriptors = []
        for img in images:
            img_uint8 = (img * 255).astype(np.uint8)  # 确保输入为uint8
            kp, desc = self.sift.detectAndCompute(img_uint8, None)
            if desc is not None:
                descriptors.extend(desc)
      
        # 仅使用部分样本加速聚类
        sample_idx = np.random.choice(len(descriptors), min(100000, len(descriptors)), replace=False)
        self.kmeans.fit(np.array(descriptors)[sample_idx])
        self.is_vocab_built = True

    def extract(self, image):
        """提取词袋特征"""
        img_uint8 = (image * 255).astype(np.uint8)
        kp, desc = self.sift.detectAndCompute(img_uint8, None)
      
        if desc is None:
            return np.zeros(self.vocab_size)
      
        visual_words = self.kmeans.predict(desc)
        hist, _ = np.histogram(visual_words, bins=self.vocab_size, range=(0, self.vocab_size))
        return hist / (hist.sum() + 1e-6)  # 归一化

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
    sift_extractor: SIFTFeatureExtractor,  # 新增参数
    sample_limit: int = 1000
) -> float:
    """评估模型准确率并记录预测结果"""
    predictions = []
    correct = 0
    limit = min(len(test_images), sample_limit)
  
    for i in range(limit):
        feature = binary_feature_extractor(test_images[i],sift_extractor)
        prediction = knn_classifier(feature, template_features, template_labels)
        predictions.append(prediction)
      
        if prediction == test_labels[i]:
            correct += 1
      
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1} samples, current accuracy: {correct/(i+1):.4f}")
  
    visualize_predictions(test_images[:limit], test_labels[:limit], predictions)
  
    return correct / limit

def binary_feature_extractor(image: np.ndarray, sift_extractor: SIFTFeatureExtractor) -> np.ndarray:
    """融合传统特征和SIFT特征"""
    # 原始特征
    thresh = threshold_otsu(image)
    binary_img = (image > thresh).astype(int)
  
    # 边缘特征
    from scipy.ndimage import sobel
    edge_x = sobel(binary_img, axis=0)
    edge_y = sobel(binary_img, axis=1)
    edge_magnitude = np.hypot(edge_x, edge_y)
  
    # 分块特征
    block_features = []
    for i in range(0, 28, 4):
        for j in range(0, 28, 4):
            block = binary_img[i:i+4, j:j+4]
            active = (block.sum() >= 8).astype(int)
            edge_block = edge_magnitude[i:i+4, j:j+4]
            edge_feature = (edge_block.sum() > 0).astype(int)
            block_features.extend([active, edge_feature])
  
    # SIFT词袋特征 (需要先构建词袋)
    sift_feature = sift_extractor.extract(image)
  
    # 特征融合
    return np.concatenate([
        np.array(block_features),
        sift_feature
    ])

def build_template_database(train_images, train_labels):
    """构建包含SIFT特征的模板库"""
    # 初始化SIFT提取器
    sift_extractor = SIFTFeatureExtractor(vocab_size=50)
  
    # 构建视觉词典
    print("Building SIFT vocabulary...")
    sift_extractor.build_vocabulary(train_images)
  
    # 提取所有特征
    print("Extracting hybrid features...")
    features = [binary_feature_extractor(img, sift_extractor) for img in train_images]
  
    # 标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(features), train_labels ,sift_extractor

def knn_classifier(test_feature, template_features, template_labels, k=5):
    """优化后的k-NN分类器"""
    # 使用余弦相似度
    similarities = np.dot(template_features, test_feature) / (
        np.linalg.norm(template_features, axis=1) * np.linalg.norm(test_feature) + 1e-6
    )
    nearest = np.argpartition(-similarities, k)[:k]
    return np.bincount(template_labels[nearest]).argmax()

def main() -> None:
    """主执行流程"""
    X_train, X_test, y_train, y_test = load_mnist_data()
    print("Dataset loaded with shapes:")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    template_db, template_labels, sift_extractor = build_template_database(X_train, y_train)
    print(f"Template database built with {len(template_db)} entries")
    
    accuracy = evaluate_accuracy(X_test, y_test, template_db, template_labels,sift_extractor)
    print(f"\nFinal test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()