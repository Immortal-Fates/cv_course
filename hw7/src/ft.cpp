/**
 *******************************************************************************
 * @file      : ft.cpp
 * @brief     :
 * @history   :
 *  Version     Date            Author          Note
 *  V0.9.0      yyyy-mm-dd      Immortal-Fates        1. <note>
 *******************************************************************************
 * @attention :
 *******************************************************************************
 *  Copyright (c) 2025 Hello World Team, Zhejiang University.
 *  All Rights Reserved.
 *******************************************************************************
 */
/* Includes ------------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* External variables --------------------------------------------------------*/
/* Exported function prototypes ----------------------------------------------*/

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// 快速傅里叶变换（图像预处理和频谱计算）
void computeDFT(Mat& image, Mat& spectrum)
{
    // 扩展图像到最佳尺寸
    Mat padded;
    int m = getOptimalDFTSize(image.rows);
    int n = getOptimalDFTSize(image.cols);
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));

    // 为实部和虚部分配空间
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexImg;
    merge(planes, 2, complexImg);

    // 傅里叶变换
    dft(complexImg, complexImg);

    // 获得幅度谱和相位谱
    split(complexImg, planes);
    magnitude(planes[0], planes[1], spectrum);
}

// 显示频谱（对数变换）
void showSpectrum(Mat& spectrum, const string& name)
{
    Mat logSpectrum;
    spectrum += Scalar::all(1);  // 避免log(0)
    log(spectrum, logSpectrum);

    // 归一化显示
    normalize(logSpectrum, logSpectrum, 0, 1, NORM_MINMAX);
    imshow(name, logSpectrum);
}

// 生成混合频谱（幅度A + 相位B）
Mat combineSpectrum(Mat& magA, Mat& phaB)
{
    Mat real, imag;
    polarToCart(magA, phaB, real, imag);

    Mat combined;
    vector<Mat> planes = {real, imag};
    merge(planes, combined);
    return combined;
}

// 逆傅里叶变换
Mat inverseDFT(Mat& complexImg)
{
    Mat inverseTransform;
    idft(complexImg, inverseTransform, DFT_SCALE | DFT_REAL_OUTPUT);
    normalize(inverseTransform, inverseTransform, 0, 1, NORM_MINMAX);
    return inverseTransform;
}

// 创建高斯低通滤波器
Mat createLowPassFilter(Size size, double D0)
{
    Mat filter = Mat::zeros(size, CV_32F);
    Point center = Point(filter.cols / 2, filter.rows / 2);

    for (int i = 0; i < filter.rows; i++) {
        for (int j = 0; j < filter.cols; j++) {
            double d = sqrt(pow(i - center.y, 2) + pow(j - center.x, 2));
            filter.at<float>(i, j) = exp(-(d * d) / (2 * D0 * D0));
        }
    }
    return filter;
}

int main()
{
    // 1. 读取图像并统一尺寸
    Mat imgA = imread("../assets/imgA.jpg", IMREAD_GRAYSCALE);
    Mat imgB = imread("../assets/imgB.jpg", IMREAD_GRAYSCALE);
    resize(imgB, imgB, imgA.size());

    // 2. 计算傅里叶变换
    Mat magA, magB, phaA, phaB;
    computeDFT(imgA, magA);
    computeDFT(imgB, magB);

    // 显示频谱
    showSpectrum(magA, "Magnitude A");
    showSpectrum(magB, "Magnitude B");

    // 3. 生成混合图像
    Mat combinedAB = combineSpectrum(magA, phaB);
    Mat resultAB = inverseDFT(combinedAB);

    Mat combinedBA = combineSpectrum(magB, phaA);
    Mat resultBA = inverseDFT(combinedBA);

    // 显示结果
    imshow("A mag + B phase", resultAB);
    imshow("B mag + A phase", resultBA);

    // 4. 频域滤波示例
    Mat filter = createLowPassFilter(imgA.size(), 30);
    Mat filteredImg;
    multiply(magA, filter, filteredImg);

    Mat inverseFiltered = inverseDFT(filteredImg);
    imshow("Low Pass Filtered", inverseFiltered);

    waitKey();
    return 0;
}