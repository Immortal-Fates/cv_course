
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

class HarrisCornerDetector
{
   public:
    /// @param ksize : sobel算子大小——必须为奇数
    /// @param k : Harris角点检测参数——常为0.04~0.06
    HarrisCornerDetector(int ksize, double k, int threshold_offset = 0)
        : ksize_(ksize), k_(k), threshold_offset_(threshold_offset) {}

    /// @brief Harris角点检测
    /// @param src : 输入图像
    /// @param corners : 输出角点
    void detect(const Mat& src, vector<Point>& corners)
    {
        // 使用sobel求x,y方向的梯度
        Mat gray;
        cvtColor(src, gray, COLOR_BGR2GRAY);
        Mat Ix, Iy;
        Sobel(gray, Ix, CV_64F, 1, 0, ksize_);
        Sobel(gray, Iy, CV_64F, 0, 1, ksize_);

        Mat Ix2 = Ix.mul(Ix);
        Mat Iy2 = Iy.mul(Iy);
        Mat Ixy = Ix.mul(Iy);

        Mat Sx2, Sy2, Sxy;
        GaussianBlur(Ix2, Sx2, Size(ksize_, ksize_), 0);
        GaussianBlur(Iy2, Sy2, Size(ksize_, ksize_), 0);
        GaussianBlur(Ixy, Sxy, Size(ksize_, ksize_), 0);

        // 只需要M矩阵的两个特征值即可
        Mat detM = Sx2.mul(Sy2) - Sxy.mul(Sxy);
        Mat traceM = Sx2 + Sy2;
        // 多种R，这里选择一种
        // R = det(M) - k * trace(M)^2
        Mat R = detM - k_ * traceM.mul(traceM);
        // R = \lambda1 * \lambda2 / (\lambda1 + \lambda2)

        normalize(R, R, 0, 255, NORM_MINMAX, CV_32FC1);
        convertScaleAbs(R, R);
        cout << "R type: " << R.type() << endl;

        // 计算自适应阈值
        Scalar mean, stddev;
        meanStdDev(R, mean, stddev);
        threshold_ = static_cast<int>(mean[0] + stddev[0]) + threshold_offset_;
        cout << "threshold = " << threshold_ << endl;

        // 显示R图
        Mat R_color;
        applyColorMap(R, R_color, COLORMAP_JET);
        namedWindow("Harris Response", WINDOW_AUTOSIZE);
        imshow("Harris Response", R_color);

        // 提取角点
        corners.clear();
        nonMaxSuppression(R, corners);
    }

    void drawCorners(Mat& image, const vector<Point>& corners)
    {
        for (const auto& corner : corners) {
            circle(image, corner, 3, Scalar(0, 255, 0), 1);
        }
    }
    void nonMaxSuppression(const Mat& R, vector<Point>& corners)
    {
        for (int i = 2; i < R.rows - 2; i++) {
            for (int j = 2; j < R.cols - 2; j++) {
                uint8_t value = R.at<uint8_t>(i, j);
                bool isLocalMax = true;
                // 5*5 window
                for (int k = -2; k <= 2; k++) {
                    for (int l = -2; l <= 2; l++) {
                        if (k == 0 && l == 0)
                            continue;
                        if (R.at<uint8_t>(i + k, j + l) >= value) {
                            isLocalMax = false;
                            break;
                        }
                    }
                    if (!isLocalMax)
                        break;
                }
                if (isLocalMax && value > threshold_) {
                    // cout << "corner value = " << static_cast<int>(value) << endl;
                    corners.push_back(Point(j, i));
                }
            }
        }
    }

   private:
    int ksize_;
    double k_;
    int threshold_;
    int threshold_offset_;
};

int main()
{
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera" << endl;
        return -1;
    }

    namedWindow("Harris Corner Detection", WINDOW_AUTOSIZE);

    HarrisCornerDetector detector(3, 0.04, 10);

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        vector<Point> corners;
        detector.detect(frame, corners);

        Mat result;
        frame.copyTo(result);
        detector.drawCorners(result, corners);

        imshow("Harris Corner Detection", result);

        if (cv::waitKey(1) == 'q') {  // 按下 'q' 键退出录制
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}