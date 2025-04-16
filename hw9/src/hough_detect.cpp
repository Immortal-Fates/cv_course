#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    Mat src = imread("../assets/imgC.png");
    if (src.empty()) {
        cout << "图像加载失败，请检查文件路径" << endl;
        return -1;
    }

    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    Mat edges;
    Canny(gray, edges, 100, 250); // 调整阈值控制边缘检测灵敏度

    vector<Vec4i> lines; 
    HoughLinesP(edges, lines, 
        1,                  // 像素分辨率rho（1像素）
        CV_PI/180,          // 角度分辨率theta（1度）
        100,                 // 累加器阈值（检测阈值）
        50,                 // 最小线段长度（像素）
        10);                // 最大线段间隙（像素）

    Mat result = src.clone();
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        line(result, 
            Point(l[0], l[1]),  
            Point(l[2], l[3]),  
            Scalar(0, 0, 255),  
            2,                  
            LINE_AA);           
    }

    imshow("原始图像", src);
    imshow("边缘检测", edges);
    imshow("直线检测结果", result);
    imwrite("../assets/result/hough_resultC.jpg", result);

    waitKey(0);
    return 0;
}