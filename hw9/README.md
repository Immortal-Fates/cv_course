
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

