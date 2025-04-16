#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    Mat src = imread("../assets/imgC.png", IMREAD_COLOR);
    if (src.empty()) {
        cout << "无法加载图像，请检查文件路径" << endl;
        return -1;
    }

    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    Mat blurred;
    GaussianBlur(gray, blurred, Size(5, 5), 1.5);
    Mat edges;
    double lowThreshold = 50;   // 低阈值
    double highThreshold = 120;  // 高阈值
    Canny(blurred, edges, lowThreshold, highThreshold);

    imshow("原始图像", src);
    imshow("边缘检测结果", edges);
  
    imwrite("../assets/result/edgesC.jpg", edges);

    waitKey(0);
    return 0;
}