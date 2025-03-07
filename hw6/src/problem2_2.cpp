#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;

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
    // 读取图像（灰度图）
    Mat image = imread("../assets/class.png");
    if (image.empty()) {
        printf("Error: Image not found!\n");
        return -1;
    }

    // 转换到HSV颜色空间
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    // 分离HSV通道
    std::vector<Mat> hsvChannels;
    split(hsv, hsvChannels);

    // 仅对亮度（V通道）进行均衡化，保留色调（H）和饱和度（S）不变，避免颜色失真。
    equalizeHist(hsvChannels[2], hsvChannels[2]);

    // 合并通道并转回BGR
    merge(hsvChannels, hsv);
    Mat result;
    cvtColor(hsv, result, COLOR_HSV2BGR);

    // 显示结果
    imshow("Original", image);
    imshow("Equalized (HSV)", result);

    imwrite("../assets/equalized_2.jpg", result);  // 保存为JPG格式
    // 显示直方图对比
    auto origin_hist = GetHistogram(image);
    auto equalized_hist = GetHistogram(result);
    imshow("Original Histogram", origin_hist);
    imshow("Equalized Histogram", equalized_hist);
    imwrite("../assets/original_histogram2.jpg", origin_hist);      // 保存为JPG格式
    imwrite("../assets/equalized_histogram2.jpg", equalized_hist);  // 保存为JPG格式

    waitKey(0);
    destroyAllWindows();

    return 0;
}