import torch
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
import cv2


# 下载训练集
# transforms.ToTensor将尺寸为[H*W*C]且位于(0,255)的所有PIL图片或者np.uint8的Numpy数组转化为尺寸为(C*H*W)且位于(0.0,1.0)的Tensor
train_dataset = datasets.MNIST(root='./dataset/MNIST/',
                train=True,
                transform=transforms.ToTensor(),
                download=False)
# 下载测试集
test_dataset = datasets.MNIST(root='./dataset/MNIST/',
               train=False,
               transform=transforms.ToTensor(),
               download=False)
 
# 装载训练集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                      batch_size=64,
                      shuffle=True)
# 装载测试集
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                     batch_size=64,
                     shuffle=True)
# [batch_size,channels,height,weight]
images, labels = next(iter(train_loader))
img = torchvision.utils.make_grid(images) 
img = img.numpy().transpose(1, 2, 0)
img = img*255
label=list(labels)

for i in range(len(label)):
    print(label[i],end="\t")
    if (i+1)%8==0:
        print()
        
cv2.imwrite('1.png', img)
