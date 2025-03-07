
#include <ctime>
#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;

int main()
{
    // 读取视频文件
    VideoCapture cap("../assets/video1.mp4");
    if (!cap.isOpened()) {
        printf("Error: Could not open video!\n");
        return -1;
    }

    // 逐帧播放
    Mat frame;
    while (cap.read(frame)) {
        imshow("Video", frame);
        if (waitKey(25) == 'q')
            break;  // 按 q 键退出
    }

    cap.release();
    destroyAllWindows();
    return 0;
}