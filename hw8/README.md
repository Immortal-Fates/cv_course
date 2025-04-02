
# 问题描述
1.   利用canny算子对指定图像进行边缘检测。

2.   利用hough变换对指定图像进行直线检测**。**

3.   尝试利用相似性原理相关的分割方法对指定图像进行分割。

（1-3）中选择两种进行编程实现，并上交实验报告



![image-20250402212904279](markdown-img/README.assets/image-20250402212904279.png)



![image-20250402212906963](markdown-img/README.assets/image-20250402212906963.png)

![image-20250402212910485](markdown-img/README.assets/image-20250402212910485.png)









# 目录结构

```
.
├── assets
│   ├── imgA_freqHP.jpg
│   ├── imgA_freqLP.jpg
│   ├── imgA.jpg
│   ├── imgA_spatialHP.jpg
│   ├── imgA_spatialLP.jpg
│   ├── imgB_freqHP.jpg
│   ├── imgB_freqLP.jpg
│   ├── imgB.jpg
│   ├── imgB_spatialHP.jpg
│   ├── imgB_spatialLP.jpg
│   ├── magnitudeA.jpg
│   ├── magnitudeB.jpg
│   ├── phaseA.jpg
│   ├── phaseB.jpg
│   ├── resultAB.jpg
│   └── resultBA.jpg
├── build
│   ├── filter.exe
│   ├── ft.exe
├── CMakeLists.txt
├── docs
│   ├── 实验报告HW2.pdf
├── README.md
└── src
    ├── filter.cpp
    └── ft.cpp
```

