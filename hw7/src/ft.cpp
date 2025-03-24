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
void computeDFT(Mat& image, Mat& magnitude, Mat& phase) {
    Mat padded;
    int m = getOptimalDFTSize(image.rows);
    int n = getOptimalDFTSize(image.cols);
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexImg;
    merge(planes, 2, complexImg);

    dft(complexImg, complexImg);

    split(complexImg, planes);

    // 正确计算幅度和相位
    magnitude.create(planes[0].size(), CV_32F);
    phase.create(planes[0].size(), CV_32F);
    cv::cartToPolar(planes[0], planes[1], magnitude, phase);

    // 转换为浮点型
    magnitude.convertTo(magnitude, CV_32F);
    phase.convertTo(phase, CV_32F);
}

// 显示并保存频谱（对数变换）
void showAndSaveSpectrum(Mat& spectrum, const string& winName, const string& filename) {
    Mat logSpectrum;
    spectrum += Scalar::all(1);  // 避免log(0)
    log(spectrum, logSpectrum);
  
    // 归一化并转换为8位格式
    normalize(logSpectrum, logSpectrum, 0, 1, NORM_MINMAX);
    Mat logSpectrum_8U;
    logSpectrum.convertTo(logSpectrum_8U, CV_8U, 255);
  
    imshow(winName, logSpectrum);
    imwrite(filename, logSpectrum_8U);
}

// 保存相位谱图像
void savePhaseImage(Mat& phase, const string& filename) {
    Mat phaseNormalized;
    normalize(phase, phaseNormalized, 0, 255, NORM_MINMAX, CV_8U);
    imwrite(filename, phaseNormalized);
}

// 生成混合频谱（幅度A + 相位B）
Mat combineSpectrum(Mat& magA, Mat& phaB) {
    CV_Assert(magA.type() == CV_32F);
    CV_Assert(phaB.type() == CV_32F);

    Mat real, imag;
    polarToCart(magA, phaB, real, imag);

    Mat combined;
    vector<Mat> planes = {real, imag};
    merge(planes, combined);
    return combined;
}

// 逆傅里叶变换并保存结果
Mat inverseDFTAndSave(Mat& complexImg, const string& filename) {
    Mat inverseTransform;
    idft(complexImg, inverseTransform, DFT_SCALE | DFT_REAL_OUTPUT);
  
    // 归一化并转换为8位格式
    Mat result;
    normalize(inverseTransform, result, 0, 1, NORM_MINMAX);
    Mat result_8U;
    result.convertTo(result_8U, CV_8U, 255);
  
    imshow(filename, result);
    imwrite(filename, result_8U);
  
    return result;
}

int main()
{
    Mat imgA = imread("../assets/imgA.jpg", IMREAD_GRAYSCALE);
    Mat imgB = imread("../assets/imgB.jpg", IMREAD_GRAYSCALE);
    resize(imgB, imgB, imgA.size());

    // 初始化幅度和相位矩阵
    Mat magA, phaA, magB, phaB;
    computeDFT(imgA, magA, phaA);
    computeDFT(imgB, magB, phaB);

    // 显示并保存幅频图像
    showAndSaveSpectrum(magA, "Magnitude A", "../assets/magnitudeA.jpg");
    showAndSaveSpectrum(magB, "Magnitude B", "../assets/magnitudeB.jpg");
  
    // 保存相位图像
    savePhaseImage(phaA, "../assets/phaseA.jpg");
    savePhaseImage(phaB, "../assets/phaseB.jpg");

    // 生成并保存混合图像
    Mat combinedAB = combineSpectrum(magA, phaB);
    Mat resultAB = inverseDFTAndSave(combinedAB, "../assets/resultAB.jpg");

    Mat combinedBA = combineSpectrum(magB, phaA);
    Mat resultBA = inverseDFTAndSave(combinedBA, "../assets/resultBA.jpg");

    waitKey();
    return 0;
}