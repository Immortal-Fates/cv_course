#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

// 四象限交换
void fftShift(Mat& mag) {
    int cx = mag.cols / 2;
    int cy = mag.rows / 2;
    Mat tmp;
    Mat q0(mag, Rect(0, 0, cx, cy));
    Mat q1(mag, Rect(cx, 0, cx, cy));
    Mat q2(mag, Rect(0, cy, cx, cy));
    Mat q3(mag, Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

// 生成运动模糊核（直接生成频率域）
Mat generateMotionBlurKernel(Size size, int len, double angle_deg) {
    Mat kernel = Mat::zeros(size, CV_32F);
    Point center(size.width / 2, size.height / 2);
    double angle_rad = angle_deg * CV_PI / 180.0;

    int dx = static_cast<int>(round((len/2) * cos(angle_rad)));
    int dy = static_cast<int>(round((len/2) * sin(angle_rad)));
    Point end1(center.x - dx, center.y - dy);
    Point end2(center.x + dx, center.y + dy);
    line(kernel, end1, end2, Scalar(1.0/len), 1);

    fftShift(kernel); 
    Mat planesH[] = { Mat_<float>(kernel.clone()), Mat::zeros(kernel.size(), CV_32F) };
    Mat H;
    merge(planesH, 2, H);
    dft(H, H, DFT_COMPLEX_OUTPUT);

    return H;
}

// 生成大气湍流传递函数
Mat generateAtmosphericTurbulenceKernel(Size size, double k) {
    Mat H_real = Mat::zeros(size, CV_32F);
    int cx = size.width / 2;
    int cy = size.height / 2;

    for (int i = 0; i < size.height; ++i) {
        for (int j = 0; j < size.width; ++j) {
            float u = j - cx;
            float v = i - cy;
            float D = sqrt(u*u + v*v);
            H_real.at<float>(i, j) = exp(-k * pow(D, 5.0/3));
        }
    }
    fftShift(H_real); // 确保原点在左上角，与DFT结果对齐

    Mat H_imag = Mat::zeros(H_real.size(), CV_32F);
    Mat planesH[] = {H_real, H_imag};
    Mat H;
    merge(planesH, 2, H);

    return H;
}

/// @brief 约束最小二乘滤波
/// @param complexDegraded 
/// @param H 
/// @param gamma 
/// @return 
Mat constrainedLeastSquaresFilter(const Mat& complexDegraded, const Mat& H, double gamma) {
    Mat laplacianKernel = (Mat_<float>(3,3) << 0, 1, 0,
                                             1, -4, 1,
                                             0, 1, 0);
    Mat paddedLaplacian = Mat::zeros(H.size(), CV_32F);
    laplacianKernel.copyTo(paddedLaplacian(Rect(0, 0, 3, 3)));
    fftShift(paddedLaplacian);

    Mat planesL[] = {paddedLaplacian, Mat::zeros(paddedLaplacian.size(), CV_32F)};
    Mat L;
    merge(planesL, 2, L);
    dft(L, L, DFT_COMPLEX_OUTPUT);

    Mat H_planes[2], L_planes[2];
    split(H, H_planes);
    split(L, L_planes);

    // |H|² + gamma*|L|²
    Mat H_mag_sq = H_planes[0].mul(H_planes[0]) + H_planes[1].mul(H_planes[1]);
    Mat L_mag_sq = L_planes[0].mul(L_planes[0]) + L_planes[1].mul(L_planes[1]);
    Mat denominator = H_mag_sq + gamma * L_mag_sq;
    denominator = max(denominator, 1e-6f);

    // F_hat = (H* * G) / (|H|² + gamma|L|²)
    Mat numerator;
    mulSpectrums(complexDegraded, H, numerator, 0, true);

    Mat F_hat_planes[2];
    split(numerator, F_hat_planes);
    divide(F_hat_planes[0], denominator, F_hat_planes[0]);
    divide(F_hat_planes[1], denominator, F_hat_planes[1]);

    Mat F_hat;
    merge(F_hat_planes, 2, F_hat);
    idft(F_hat, F_hat, DFT_SCALE | DFT_REAL_OUTPUT);

    Mat restored;
    normalize(F_hat, restored, 0, 1, NORM_MINMAX);
    return restored;
}

/// @brief 维纳滤波
/// @param complexDegraded 
/// @param H 
/// @param K 
/// @return 
Mat wienerFilter(const Mat& complexDegraded, const Mat& H, double K) {
    Mat H_planes[2];
    split(H, H_planes);

    Mat H_conj_planes[] = {H_planes[0], -H_planes[1]};
    Mat H_conj;
    merge(H_conj_planes, 2, H_conj);
  
    Mat H_mag_abs;
    magnitude(H_planes[0], H_planes[1], H_mag_abs);
    Mat H_mag_sq = H_mag_abs.mul(H_mag_abs);

    Mat numerator;
    mulSpectrums(complexDegraded, H_conj, numerator, 0, true);
    // cout << "numerator:" << numerator << endl;

    Mat denominator;
    // Mat denom_planes[] = {H_mag_sq + K, Mat::zeros(H_mag_sq.size(), CV_32F)};
    // 这里不是复数除法，只有实数部分，但是divide是分别相除，因此两个部分都设置为H_mag_sq + K
    Mat denom_planes[] = {H_mag_sq + K, H_mag_sq + K};
    merge(denom_planes, 2, denominator);

    Mat F_hat;
    divide(numerator, denominator, F_hat);

    Mat test[2];
    split(F_hat, test);

    Mat mag, logMag;
    magnitude(test[0], test[1], mag);
    fftShift(mag);
    mag += Scalar::all(1);
    log(mag, logMag);

    normalize(logMag, logMag, 0, 255, NORM_MINMAX);
    Mat spectrum;
    logMag.convertTo(spectrum, CV_8UC1);
    imshow("Spectrum of test[0]", spectrum);

    merge(test, 2, F_hat);
    idft(F_hat, F_hat, DFT_SCALE);

    Mat planes[2];
    split(F_hat, planes);
    Mat restored;
    normalize(planes[0], restored, 0, 1, NORM_MINMAX);

    return restored;
}

int main() {
    Mat img = imread("../assets/imgB.jpg");
    if (img.empty()) {
        cout << "图像加载失败！" << endl;
        return -1;
    }

    Mat grayImg;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);
    grayImg.convertTo(grayImg, CV_32F);

    int M = getOptimalDFTSize(grayImg.rows);
    int N = getOptimalDFTSize(grayImg.cols);
    Mat padded;
    copyMakeBorder(grayImg, padded, 0, M-grayImg.rows, 0, N-grayImg.cols, BORDER_REPLICATE); // 改用BORDER_REPLICATE减少边缘效应

    Mat H;
    int choice;
    cout << "选择退化模型 (1: 运动模糊, 2: 大气湍流): ";
    cin >> choice;

    if (choice == 1) {
        int len = 20;
        double angle = 0;
        H = generateMotionBlurKernel(padded.size(), len, angle);
    } else if (choice == 2) {
        double k = 0.0025;
        H = generateAtmosphericTurbulenceKernel(padded.size(), k);
    } else {
        cout << "无效选择！" << endl;
        return -1;
    }

    Mat planes[] = {padded, Mat::zeros(padded.size(), CV_32F)};
    Mat complexImg;
    merge(planes, 2, complexImg);
    dft(complexImg, complexImg);

    mulSpectrums(complexImg, H, complexImg, 0);

    idft(complexImg, complexImg, DFT_SCALE);
    split(complexImg, planes);
    Mat degraded;
    normalize(planes[0], degraded, 0, 1, NORM_MINMAX);

    Mat planesDegraded[] = {degraded, Mat::zeros(degraded.size(), CV_32F)};
    Mat complexDegraded;
    merge(planesDegraded, 2, complexDegraded);
    dft(complexDegraded, complexDegraded);

    int method;
    cout << "\n选择复原方法 (1: 维纳滤波, 2: 约束最小二乘): ";
    cin >> method;

    Mat restored;
    switch (method) {
        case 1: {
            double K = 0.01;
            restored = wienerFilter(complexDegraded, H, K);
            break;
        }
        case 2: {
            double gamma = 0.001;  // 需要调整的参数
            restored = constrainedLeastSquaresFilter(complexDegraded, H, gamma);
            break;
        }
        default:
            cout << "无效选择，使用默认维纳滤波" << endl;
            restored = wienerFilter(complexDegraded, H, 0.01);
    }

    Mat degradedDisplay, restoredDisplay;
    degraded.convertTo(degradedDisplay, CV_8U, 255);
    restored.convertTo(restoredDisplay, CV_8U, 255);
    
    string modelName;
    switch (choice) {
        case 1: modelName = "motion"; break;
        case 2: modelName = "atmospheric"; break;
        default: modelName = "unknown";
    }
    string methodName;
    switch (method) {
        case 1: methodName = "wiener"; break;
        case 2: methodName = "clsq"; break;
        default: methodName = "default";
    }
    string prefix = modelName + "_" + methodName;
    string degradedPath = "../assets/results/" + prefix + "_degraded.png";
    string restoredPath = "../assets/results/" + prefix + "_restored.png";

    imwrite(degradedPath, degradedDisplay);
    imwrite(restoredPath, restoredDisplay);

    Mat displayGray_1;
    normalize(grayImg, displayGray_1, 0.0, 1.0, NORM_MINMAX);
    imshow("原图", displayGray_1);
    imshow("退化图像", degradedDisplay);
    imshow("复原图像", restoredDisplay);
    waitKey(0);

    return 0;
}