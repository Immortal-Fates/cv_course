#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

// 生成高斯低通滤波器
Mat createGaussianLowPassFilter(Size size, double D0) {
    Mat filter = Mat::zeros(size, CV_32F);
    Point center = Point(size.width/2, size.height/2);
  
    for(int i = 0; i < size.height; i++) {
        for(int j = 0; j < size.width; j++) {
            double d = sqrt(pow(i - center.y, 2) + pow(j - center.x, 2));
            filter.at<float>(i, j) = exp(-(d*d)/(2*D0*D0));
        }
    }
    return filter;
}

// 生成理想高通滤波器
Mat createIdealHighPassFilter(Size size, int D0) {
    Mat filter = Mat::ones(size, CV_32F);
    Point center = Point(size.width/2, size.height/2);
  
    for(int i = 0; i < size.height; i++) {
        for(int j = 0; j < size.width; j++) {
            if(sqrt(pow(i - center.y, 2) + pow(j - center.x, 2)) < D0) {
                filter.at<float>(i, j) = 0;
            }
        }
    }
    return filter;
}

// 频域滤波处理
Mat frequencyDomainFilter(Mat& src, Mat& filter) {
    // 扩展图像尺寸
    Mat padded;
    int m = getOptimalDFTSize(src.rows);
    int n = getOptimalDFTSize(src.cols);
    copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));
  
    // 转换为浮点型
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexImg;
    merge(planes, 2, complexImg);
  
    // 傅里叶变换
    dft(complexImg, complexImg);
  
    // 应用滤波器
    Mat filterPadded;
    resize(filter, filterPadded, padded.size());
    Mat filterPlanes[] = {filterPadded, filterPadded};
    Mat filterComplex;
    merge(filterPlanes, 2, filterComplex);
  
    mulSpectrums(complexImg, filterComplex, complexImg, 0);
  
    // 逆变换
    idft(complexImg, complexImg, DFT_SCALE | DFT_REAL_OUTPUT);
  
    // 裁剪结果
    Mat result;
    complexImg(Rect(0, 0, src.cols, src.rows)).copyTo(result);
    normalize(result, result, 0, 1, NORM_MINMAX);
    return result;
}

int main() {
    // 读取多张测试图像
    vector<string> imgPaths = {
        "../assets/imgA.jpg",  
        "../assets/imgB.jpg",  
    };
  
    for(auto& path : imgPaths) {
        Mat src = imread(path, IMREAD_GRAYSCALE);
        if(src.empty()) {
            cerr << "无法读取图像: " << path << endl;
            continue;
        }
      
        /**************** 空域滤波 ****************/
        // 高斯低通滤波（模糊）
        Mat spatialLowPass;
        GaussianBlur(src, spatialLowPass, Size(15,15), 3);
      
        // 高通滤波（拉普拉斯边缘检测）
        Mat spatialHighPass;
        Laplacian(src, spatialHighPass, CV_32F, 3);
        convertScaleAbs(spatialHighPass, spatialHighPass);
      
        /**************** 频域滤波 ****************/
        // 生成滤波器
        Mat gaussianLPF = createGaussianLowPassFilter(src.size(), 30);
        Mat idealHPF = createIdealHighPassFilter(src.size(), 30);
      
        // 应用频域滤波
        Mat freqLowPass = frequencyDomainFilter(src, gaussianLPF);
        Mat freqHighPass = frequencyDomainFilter(src, idealHPF);
      
        /**************** 保存结果 ****************/
        string prefix = path.substr(0, path.find_last_of('.'));
      
        imwrite(prefix + "_spatialLP.jpg", spatialLowPass);
        imwrite(prefix + "_spatialHP.jpg", spatialHighPass);
        imwrite(prefix + "_freqLP.jpg", freqLowPass*255);
        imwrite(prefix + "_freqHP.jpg", freqHighPass*255);
      
        /**************** 显示结果 ****************/
        imshow("Original", src);
        imshow("Spatial LowPass", spatialLowPass);
        imshow("Spatial HighPass", spatialHighPass);
        imshow("Frequency LowPass", freqLowPass);
        imshow("Frequency HighPass", freqHighPass);
    }
  
    waitKey();
    return 0;
}