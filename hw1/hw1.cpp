#include <ctime>
#include <opencv2/opencv.hpp>
#include <string>

int main()
{
    // 初始化摄像头
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }

    // 获取当前时间用于生成文件名
    time_t now = time(0);
    tm *ltm = localtime(&now);
    char current_time[20];
    strftime(current_time, sizeof(current_time), "%Y-%m-%d_%H-%M-%S", ltm);
    std::string output_file = "output_video_" + std::string(current_time) + ".mp4";

    // 创建视频写入器
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::Size frameSize(frame_width, frame_height);             // 定义frameSize变量
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');  // 使用'mp4v'编解码器
    cv::VideoWriter out(output_file, fourcc, 20.0, frameSize);

    // 添加字幕的字体
    int font = cv::FONT_HERSHEY_SIMPLEX;  // 使用int类型代替FontFace

    // 添加图标

    cv::Mat logo = cv::imread("../img/logo.jpeg");
    if (logo.empty()) {
        std::cerr << "Error: Could not load logo image" << std::endl;
        return -1;
    }

    // 获取用户姓名
    std::string username = "yxy";

    // 调整图标图像尺寸，使其更小
    cv::resize(logo, logo, cv::Size(100, 100));

    // 视频录制循环
    while (true) {
        // 获取摄像头帧
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // 添加时间显示（右上角）
        now = time(0);
        ltm = localtime(&now);
        strftime(current_time, sizeof(current_time), "%H:%M:%S", ltm);
        cv::putText(frame, current_time, cv::Point(frame.cols - 150, 20), font, 1, cv::Scalar(255, 255, 255), 2);

        // 添加字幕（底部）
        std::string subtitle = "Hello OpenCV";
        cv::putText(frame, subtitle, cv::Point(50, frame.rows - 30), font, 1, cv::Scalar(255, 255, 255), 2);

        // 添加图标和用户名（左上角）
        logo.copyTo(frame(cv::Rect(0, 0, logo.cols, logo.rows)));
        cv::putText(frame, username, cv::Point(10, logo.rows + 30), font, 1, cv::Scalar(255, 255, 255), 2);

        // 将帧写入视频文件
        out.write(frame);

        // 显示帧
        cv::imshow("Camera", frame);
        if (cv::waitKey(1) == 'q') {  // 按下 'q' 键退出录制
            break;
        }
    }

    // 释放摄像头和视频写入器资源
    cap.release();
    out.release();
    cv::destroyAllWindows();

    // 打印输出文件名，提示用户生成的视频文件位置
    std::cout << "视频已保存为 " << output_file << std::endl;
    return 0;
}