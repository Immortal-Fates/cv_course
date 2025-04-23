
# 问题描述
（1和2任选一题）

1. 任选一图像，用运动模糊传递函数/大气湍流传递函数对图像进行退化，再采用所介绍的复原滤波器复原该图像。
2. 对所提供的图像（见附件），采用所介绍的复原滤波器进行复原。所提供的图像是经过长度30、逆时针方向角度为11度的运动模糊退化的图像，并加有高斯白噪声。
3. （拓展）自行尝试多种去噪和复原滤波操作。
4. 编程实现并上传实验报告（可附代码）。







# 目录结构

```
.
├── assets
│   ├── imgA.png
│   ├── imgB.png
│   ├── imgC.png
│   └── result
│       ├── edgesB.jpg
│       ├── edgesC.jpg
│       ├── edges.jpg
│       ├── hough_resultA.jpg
│       ├── hough_resultB.jpg
│       ├── hough_resultC.jpg
│       └── hough_result.jpg
├── build
│   ├── hough.exe
│   ├── img_seg.exe
├── CMakeLists.txt
├── docs
│   └── 实验报告HW3.pdf
├── README.md
└── src
    ├── canny_detect.cpp
    └── hough_detect.cpp
```

# OpenCV使用
- copyMakeBorder():Forms a border around an image.
- merge():Creates one multi-channel array out of several single-channel ones.

# Problem 
- grayImg转换为CV_32F后，数值范围变为[0.0, 255.0]，而imshow()对浮点图像的默认处理方式为：
    - 将像素值视为[0.0, 1.0]范围，超出1.0的值会被截断为白色
    - 因此当像素值>1.0时，整个图像显示为纯白色

- 四象限交换
   离散傅里叶变换（DFT）特性：DFT默认将零频分量（直流分量）放在频谱的左上角（图像四角），高频分量分布在中心区域。四象限交换通过翻转频域矩阵，将零频分量移至中心，高频分量移至四角

- 傅里叶变换对称性
    实数图像的频域具有共轭对称性，未正确对齐会导致空域图像对称伪影

