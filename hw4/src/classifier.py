import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import random_split

# 定义深度可分离卷积
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# 定义MobileNetV1模型
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV1, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(64, 128, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(128, 256, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(256, 512, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            *[DepthwiseSeparableConv(512, 512) for _ in range(5)],
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(512, 1024, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(1024, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),  # 添加Dropout层
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def get_data_loaders(batch_size, data_path='./dataset/CIFAR-10/'):
    # 数据预处理
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # 添加随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 添加颜色抖动
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 加载CIFAR-10数据集
    full_train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

    # 划分训练集和验证集
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    model.to(device)
    model.train()
    loss_list = []
    val_loss_list = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i+1) % 100 == 0:
                avg_loss = running_loss / 100
                loss_list.append(avg_loss)
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {avg_loss:.4f}')
                running_loss = 0.0
        
        # 每个epoch结束后更新学习率
        scheduler.step()

        # 验证模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_loss_list.append(val_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
        model.train()

    return loss_list, val_loss_list

def test_model(model, test_loader, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f'Accuracy of the model on the 10000 test images: {100 * correct / total} %')

def show_predictions(model, test_loader, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        fig = plt.figure(figsize=(15, 15))
        for i in range(10):
            ax = fig.add_subplot(2, 5, i+1)
            ax.imshow(images[i].cpu().numpy().transpose((1, 2, 0)))
            ax.set_title(f'True: {labels[i].item()}\nPred: {predicted[i].item()}')
            ax.axis('off')
        plt.show()

def main():
    batch_size = 128
    learning_rate = 0.01  # 进一步降低学习率
    num_epochs = 50  # 增加训练轮数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    dataset_path = './dataset/CIFAR-10/'

    train_loader, val_loader, test_loader = get_data_loaders(batch_size, dataset_path)
    model = MobileNetV1(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)  # 使用SGD优化器
    
    # 使用学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Train the model
    loss_list, val_loss_list = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)
    
    # Save the model checkpoint
    torch.save(model.state_dict(), './model/Mobile_model3.pth')
    print('Model saved to Mobile_model3.pth')

    # Load the trained model
    model.load_state_dict(torch.load('./model/Mobile_model3.pth'))
    print('Model loaded from Mobile_model3.pth')
    
    # Test the model
    test_model(model, test_loader, device)

    # Show predictions for the first 10 images in the test set
    show_predictions(model, test_loader, device)

    # Plot the loss curve
    plt.figure()
    plt.plot(loss_list, label='Training Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()