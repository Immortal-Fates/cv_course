
# 问题描述
1.	任选两幅大小一致的灰度图像，分别用A、B表示；
2.	分别对A和B做傅立叶变换，求各自的幅频和相频（用图像形式表示）；
3.	用A的幅频和B的相频进行逆傅立叶变换，生成并显示新图像；
4.	用B的幅频和A的相频进行逆傅立叶变换，生成并显示新图像，并对比分析；
5.	自行选择多张图像，对图像的空域/频域进行的高通/低通等滤波操作，分析结果；
6.	上交实验报告。



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

