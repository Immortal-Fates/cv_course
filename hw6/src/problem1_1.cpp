
#include <ctime>
#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
int main()
{
    // 读取图像
    Mat image = imread("../assets/colored.png");
    if (image.empty()) {
        printf("Error: Image not found!\n");
        return -1;
    }

    // 比例缩放（缩小50%）
    Mat resized;
    resize(image, resized, Size(), 0.5, 0.5, INTER_LINEAR);

    // 保存缩放后的图像
    imwrite("../assets/colored_resized.jpg", resized);  // 保存为JPG格式

    // 显示结果
    imshow("Original", image);
    imshow("Resized", resized);
    waitKey(0);
    destroyAllWindows();

    return 0;
}