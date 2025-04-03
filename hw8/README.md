
# 问题描述
1.   利用canny算子对指定图像进行边缘检测。

2.   利用hough变换对指定图像进行直线检测。

3.   尝试利用相似性原理相关的分割方法对指定图像进行分割。

（1-3）中选择两种进行编程实现，并上交实验报告



![image-20250402212904279](markdown-img/README.assets/image-20250402212904279.png)



![image-20250402212906963](markdown-img/README.assets/image-20250402212906963.png)

![image-20250402212910485](markdown-img/README.assets/image-20250402212910485.png)









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

