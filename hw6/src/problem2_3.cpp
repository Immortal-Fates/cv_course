#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;

void MatchHistogram(const Mat& src, const Mat& target, Mat& matched)
{
    // 计算源图像和目标图像的直方图
    Mat src_hist, target_hist;
    const float range[] = {0, 256};
    const float* ranges[] = {range};

    int channels[] = {0};
    int histSize[] = {256};

    calcHist(&src, 1, channels, Mat(), src_hist, 1, histSize, ranges);
    calcHist(&target, 1, channels, Mat(), target_hist, 1, histSize, ranges);

    // 归一化直方图
    normalize(src_hist, src_hist, 0, 1, NORM_MINMAX);
    normalize(target_hist, target_hist, 0, 1, NORM_MINMAX);

    // 计算累积分布函数 (CDF)
    Mat src_cdf = src_hist.clone(), target_cdf = target_hist.clone();
    for (int i = 1; i < 256; i++) {
        src_cdf.at<float>(i) += src_cdf.at<float>(i - 1);
        target_cdf.at<float>(i) += target_cdf.at<float>(i - 1);
    }

    // 构建映射表
    Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; i++) {
        float min_diff = 1.0;
        for (int j = 0; j < 256; j++) {
            float diff = fabs(src_cdf.at<float>(i) - target_cdf.at<float>(j));
            if (diff < min_diff) {
                min_diff = diff;
                lut.at<uchar>(i) = j;
            }
        }
    }

    // 应用映射表
    LUT(src, lut, matched);
}

// 显示图像和直方图
Mat GetHistogram(const Mat& image)
{
    // 计算直方图
    Mat hist;
    int channels[] = {0};
    int histSize[] = {256};
    float range[] = {0, 256};
    const float* ranges[] = {range};
    calcHist(&image, 1, channels, Mat(), hist, 1, histSize, ranges);

    // 归一化直方图以便显示
    normalize(hist, hist, 0, 400, NORM_MINMAX);  // 400为显示高度

    // 创建直方图画布
    Mat histImage(400, 512, CV_8UC3, Scalar(255, 255, 255));

    // 绘制直方图
    for (int i = 0; i < 256; i++) {
        rectangle(histImage, Point(i * 2, 400 - cvRound(hist.at<float>(i))),
                  Point((i + 1) * 2, 400), Scalar(0, 0, 255), -1);
    }

    return histImage;
}

int main()
{
    // 读取源图像和目标图像（灰度图）
    Mat src = imread("../assets/colored.png", IMREAD_GRAYSCALE);
    Mat target = imread("../assets/class.png", IMREAD_GRAYSCALE);
    if (src.empty() || target.empty()) {
        printf("Error: Images not found!\n");
        return -1;
    }

    // 直方图匹配
    Mat matched;
    MatchHistogram(src, target, matched);

    // 显示结果
    imshow("Source", src);
    imshow("Target", target);
    imshow("Matched Result", matched);

    auto src_hist = GetHistogram(src);
    auto target_hist = GetHistogram(target);
    auto matched_hist = GetHistogram(matched);

    imshow("Source Histogram", src_hist);
    imshow("Target Histogram", target_hist);
    imshow("Matched Histogram", matched_hist);

    imwrite("../assets/matched.jpg", matched);  // 保存为JPG格式
    imwrite("../assets/src.jpg", src);          // 保存为JPG格式
    imwrite("../assets/target.jpg", target);    // 保存为JPG格式

    imwrite("../assets/target_hist.jpg", target_hist);    // 保存为JPG格式
    imwrite("../assets/src_hist.jpg", src_hist);          // 保存为JPG格式
    imwrite("../assets/matched_hist.jpg", matched_hist);  // 保存为JPG格式

    waitKey(0);
    destroyAllWindows();
    return 0;
}