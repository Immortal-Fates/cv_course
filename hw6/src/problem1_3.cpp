#include <opencv2/opencv.hpp>
using namespace cv;

int main()
{
    // 读取图像
    Mat image = imread("../assets/colored.png");
    if (image.empty()) {
        printf("Error: Image not found!\n");
        return -1;
    }

    // 裁剪（取左上角1/4区域）
    Rect crop_roi(0, 0, image.cols / 2, image.rows / 2);
    Mat cropped = image(crop_roi);

    // 水平翻转
    Mat flipped;
    flip(image, flipped, 1);

    // 颜色通道转换（BGR转RGB）
    Mat rgb;
    cvtColor(image, rgb, COLOR_BGR2RGB);

    imwrite("../assets/cropped.jpg", cropped);  // 保存为JPG格式
    imwrite("../assets/flipped.jpg", flipped);  // 保存为JPG格式
    imwrite("../assets/rgb.jpg", rgb);          // 保存为JPG格式

    // 显示所有结果（直接单独显示）
    imshow("Original", image);
    imshow("Cropped", cropped);
    imshow("Flipped", flipped);

    // 注意：OpenCV默认显示BGR格式，直接显示RGB图像颜色会异常
    imshow("RGB (Incorrect)", rgb);  // 颜色异常

    // 正确显示RGB的方式：将RGB转回BGR
    Mat rgb_display;
    cvtColor(rgb, rgb_display, COLOR_RGB2BGR);
    imshow("RGB (Corrected)", rgb_display);

    waitKey(0);
    destroyAllWindows();

    return 0;
}