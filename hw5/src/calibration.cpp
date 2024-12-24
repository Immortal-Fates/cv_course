#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

class CameraCalibration
{
   public:
    CameraCalibration(const cv::Size& board_size)
        : board_size_(board_size)
    {
        // Initialize object points
        for (int j = 0; j < board_size_.height; ++j) {
            for (int i = 0; i < board_size_.width; ++i) {
                object_points_.emplace_back(i, j, 0.0f);
            }
        }
    }

    void addChessboardPoints(const std::vector<std::string>& filelist)
    {
        int display_count = 0;  // 用于计数显示的图像数量

        for (const auto& file : filelist) {
            cv::Mat image = cv::imread(file);  // 读取彩色图像
            cv::Mat gray_image;
            cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);  // 转换为灰度图像
            std::vector<cv::Point2f> corners;

            if (cv::findChessboardCorners(gray_image, board_size_, corners)) {
                cv::cornerSubPix(gray_image, corners, cv::Size(11, 11), cv::Size(-1, -1),
                                 cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
                image_points_.push_back(corners);
                object_points_list_.push_back(object_points_);

                // 显示棋盘角点检测结果
                if (display_count < 2) {
                    cv::drawChessboardCorners(image, board_size_, corners, 1);
                    std::string window_name = "Chessboard Corners " + std::to_string(display_count + 1);
                    cv::imshow(window_name, image);  // 显示彩色图像
                    display_count++;
                }
            } else {
                std::cout << "Chessboard not found in " << file << std::endl;
            }
        }
    }

    void calibrate()
    {
        std::vector<cv::Mat> rvecs, tvecs;
        cv::calibrateCamera(object_points_list_, image_points_, board_size_, camera_matrix_, dist_coeffs_, rvecs, tvecs);

        // 计算重投影误差
        total_reprojection_error_ = computeReprojectionErrors(object_points_list_, image_points_, rvecs, tvecs);
    }

    void printCoeff() const
    {
        std::cout << "Camera Matrix: " << camera_matrix_ << std::endl;
        std::cout << "Distortion Coefficients: " << dist_coeffs_ << std::endl;
        std::cout << "Total Reprojection Error: " << total_reprojection_error_ << std::endl;
    }

    void saveCoeff(const std::string& filename) const
    {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        fs << "Camera_Matrix" << camera_matrix_;
        fs << "Distortion_Coefficients" << dist_coeffs_;
        fs << "Total_Reprojection_Error" << total_reprojection_error_;
        fs.release();
    }
    void loadCoeff(const std::string& filename)
    {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        fs["Camera_Matrix"] >> camera_matrix_;
        fs["Distortion_Coefficients"] >> dist_coeffs_;
        fs.release();
    }

    void undistortAndShow(const std::vector<std::string>& filelist) const
    {
        int display_count = 0;
        for (const auto& file : filelist) {
            if (display_count >= 2)
                break;  // 只显示两张图像

            cv::Mat image = cv::imread(file);  // 读取彩色图像
            if (image.empty())
                continue;

            cv::Mat undistorted_image;
            cv::undistort(image, undistorted_image, camera_matrix_, dist_coeffs_);

            std::string window_name = "Undistorted Image " + std::to_string(display_count + 1);
            cv::imshow(window_name, undistorted_image);  // 显示校正后的图像
            display_count++;
        }
    }
    void transformToBirdsEyeView(const std::string& input_image_path, const std::string& output_image_path) const
    {
        cv::Mat image = cv::imread(input_image_path);  // 读取彩色图像
        if (image.empty()) {
            std::cerr << "Error: Unable to load image!" << std::endl;
            return;
        }

        cv::Mat gray_image;
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

        // 去畸变
        cv::Mat undistorted_image;
        cv::undistort(image, undistorted_image, camera_matrix_, dist_coeffs_);

        // 检测棋盘格角点
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(undistorted_image, board_size_, corners);
        if (!found) {
            std::cerr << "Error: Couldn't acquire checkerboard corners!" << std::endl;
            return;
        }

        // 获取亚像素级别的角点位置
        cv::cornerSubPix(gray_image, corners, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

        std::cout << "size" << undistorted_image.size() << std::endl;
        // 定义图像和对象点
        std::vector<cv::Point2f> objPts, imgPts;
        int center_x = undistorted_image.cols / 2;
        int center_y = undistorted_image.rows / 2;
        int offset_x = 180;
        int offset_y = 100;
        objPts.push_back(cv::Point2f(center_x - offset_x, center_y - offset_y));
        objPts.push_back(cv::Point2f(center_x + offset_x, center_y - offset_y));
        objPts.push_back(cv::Point2f(center_x - offset_x, center_y + offset_y));
        objPts.push_back(cv::Point2f(center_x + offset_x, center_y + offset_y));
        // objPts.push_back(cv::Point2f(300, 300));
        // objPts.push_back(cv::Point2f(660, 300));
        // objPts.push_back(cv::Point2f(300, 510));
        // objPts.push_back(cv::Point2f(660, 510));

        imgPts.push_back(corners[0]);
        imgPts.push_back(corners[board_size_.width - 1]);
        imgPts.push_back(corners[(board_size_.height - 1) * board_size_.width]);
        imgPts.push_back(corners[(board_size_.height - 1) * board_size_.width + board_size_.width - 1]);

        // 计算单应性矩阵
        cv::Mat H = cv::getPerspectiveTransform(objPts, imgPts);
        std::cout << "H:" << H << std::endl;

        // 允许用户调整视角高度
        double Z = 1.0;  // 初始高度
        cv::Mat birds_eye_view;
        cv::namedWindow("Bird's Eye View");
        int key = 0;
        while (key != 27) {  // 按 ESC 键退出
            // H.at<double>(2, 2) = Z;  // 更新单应性矩阵中的 Z 值
            int temp_offset_x = offset_x / Z;
            int temp_offset_y = offset_y / Z;
            objPts.clear();
            objPts.push_back(cv::Point2f(center_x - temp_offset_x, center_y - temp_offset_y));
            objPts.push_back(cv::Point2f(center_x + temp_offset_x, center_y - temp_offset_y));
            objPts.push_back(cv::Point2f(center_x - temp_offset_x, center_y + temp_offset_y));
            objPts.push_back(cv::Point2f(center_x + temp_offset_x, center_y + temp_offset_y));
            cv::Mat H = cv::getPerspectiveTransform(objPts, imgPts);

            cv::warpPerspective(undistorted_image, birds_eye_view, H, undistorted_image.size(),
                                cv::INTER_LINEAR + cv::WARP_INVERSE_MAP + cv::WARP_FILL_OUTLIERS);
            cv::imshow("Bird's Eye View", birds_eye_view);
            key = cv::waitKey(1);  // 等待按键
            if (key == 'u')
                Z += 0.5;  // 按 'u' 键升高视角
            if (key == 'd')
                Z -= 0.5;  // 按 'd' 键降低视角
        }

        // 保存结果
        cv::imwrite(output_image_path, birds_eye_view);
    }

   private:
    double computeReprojectionErrors(const std::vector<std::vector<cv::Point3f>>& object_points,
                                     const std::vector<std::vector<cv::Point2f>>& image_points,
                                     const std::vector<cv::Mat>& rvecs,
                                     const std::vector<cv::Mat>& tvecs) const
    {
        std::vector<cv::Point2f> image_points2;
        double total_error = 0;
        int total_points = 0;

        for (size_t i = 0; i < object_points.size(); ++i) {
            cv::projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix_, dist_coeffs_, image_points2);
            double error = cv::norm(image_points[i], image_points2, cv::NORM_L2);
            total_error += error * error;
            total_points += static_cast<int>(object_points[i].size());
        }

        return std::sqrt(total_error / total_points);
    }
    cv::Size board_size_;
    std::vector<cv::Point3f> object_points_;
    std::vector<std::vector<cv::Point2f>> image_points_;
    std::vector<std::vector<cv::Point3f>> object_points_list_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    double total_reprojection_error_;
};

int main(int argc, char** argv)
{
    // 定义棋盘格的尺寸
    cv::Size chessboard_size(9, 6);  // 9x6 内角点

    // 创建相机标定对象
    CameraCalibration calibrator(chessboard_size);

    // 读取指定目录中的所有 .bmp 图像文件
    std::string path = "../assets/calib2/";
    std::vector<std::string> filelist;
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.path().extension() == ".bmp") {
            filelist.push_back(entry.path().string());
            std::cout << "Found image: " << entry.path().string() << std::endl;
        }
    }

    // 添加棋盘格点
    calibrator.addChessboardPoints(filelist);

    // 标定相机
    calibrator.calibrate();

    // 输出相机内参和畸变系数
    calibrator.printCoeff();

    // 保存相机内参和畸变系数到指定路径的 .yaml 文件
    std::string output_path = "../assets/result/calibration_results.yaml";
    calibrator.saveCoeff(output_path);
    // 显示两张图像的畸变校正结果
    calibrator.undistortAndShow(filelist);

    calibrator.loadCoeff(output_path);
    // 输入一张平地拍摄的图片，输出俯瞰视角变换后的图片
    std::string input_image_path = "../assets/bird_eye/img2.jpeg";
    std::string output_image_path = "../assets/result/birds_eye_view.jpeg";
    calibrator.transformToBirdsEyeView(input_image_path, output_image_path);

    cv::waitKey(0);           // 等待用户按键
    cv::destroyAllWindows();  // 关闭所有显示的窗口

    return 0;
}