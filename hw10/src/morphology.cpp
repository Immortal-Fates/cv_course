#include <vector>

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

namespace {

constexpr int kDefaultImageSize = 400;
constexpr int kKernelSize = 7;
const cv::Point kShapeCenter(200, 200);
const int kShapeRadius = 80;
const cv::Rect kRectangleRegion(100, 100, 200, 200);

cv::Mat CreateTestImage() {
  cv::Mat image = cv::Mat::zeros(kDefaultImageSize, kDefaultImageSize, CV_8UC1);

  cv::rectangle(image, kRectangleRegion, cv::Scalar(255), cv::FILLED);
  cv::circle(image, kShapeCenter, kShapeRadius, cv::Scalar(0), cv::FILLED);

  return image;
}

// 创建不同形态的结构元素
std::vector<cv::Mat> CreateStructureElements() {
  const cv::Size kernel_size(kKernelSize, kKernelSize);
  return {
    cv::getStructuringElement(cv::MORPH_RECT, kernel_size),
    cv::getStructuringElement(cv::MORPH_ELLIPSE, kernel_size),
    cv::getStructuringElement(cv::MORPH_CROSS, kernel_size)
  };
}

// 执行形态学操作并返回结果
cv::Mat ApplyMorphologyOperation(const cv::Mat& src, 
                               const cv::Mat& kernel,
                               int operation_type,
                               const std::string& operation_name,
                               const std::string& kernel_name
                            ) {
  cv::Mat result;
  if (operation_type == cv::MORPH_ERODE) {
    cv::erode(src, result, kernel);
  } else if (operation_type == cv::MORPH_DILATE) {
    cv::dilate(src, result, kernel);
  } else {
    cv::morphologyEx(src, result, operation_type, kernel);
  }

  cv::Mat color_result;
  cv::cvtColor(result, color_result, cv::COLOR_GRAY2BGR);
  const std::string text = operation_name + " - " + kernel_name;
  cv::putText(color_result, text, cv::Point(10, 30),
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

  return color_result;
}

// 生成单行比较结果
cv::Mat GenerateMorphologyRow(const cv::Mat& src, 
                             const cv::Mat& kernel,
                             const std::string& kernel_name) {
  std::vector<cv::Mat> operations;

  const auto add_operation = [&](int type, const std::string& name,const std::string& kernel_name) {
    operations.push_back(ApplyMorphologyOperation(src, kernel, type, name, kernel_name));
  };

  add_operation(cv::MORPH_ERODE, "Erode", kernel_name);
  add_operation(cv::MORPH_DILATE, "Dilate", kernel_name);
  add_operation(cv::MORPH_OPEN, "Open", kernel_name);
  add_operation(cv::MORPH_CLOSE, "Close", kernel_name);

  cv::Mat row_result;
  cv::hconcat(operations, row_result);
  return row_result;
}

// 细化操作（实现Zhang-Suen算法）
cv::Mat ThinImage(const cv::Mat& src) {
    cv::Mat dst;
    cv::ximgproc::thinning(src, dst, cv::ximgproc::THINNING_ZHANGSUEN);
    return dst;
}

// 骨架提取（通过迭代腐蚀）
cv::Mat Skeletonize(const cv::Mat& src) {
    cv::Mat skel(src.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp, eroded;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

    bool done;
    do {
        cv::erode(src, eroded, element);
        cv::dilate(eroded, temp, element);
        cv::subtract(src, temp, temp);
        cv::bitwise_or(skel, temp, skel);
        eroded.copyTo(src);
        done = (cv::countNonZero(src) == 0);
    } while (!done);

    return skel;
}

// 孔洞填充
cv::Mat morphReconstructFill(const cv::Mat& src) {
    CV_Assert(src.type() == CV_8UC1);
    cv::Mat inverted;
    cv::bitwise_not(src, inverted);
    cv::Mat marker = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::rectangle(marker, cv::Rect(1,1,src.cols-2,src.rows-2), 255, cv::FILLED);
    cv::Mat prev;
    do {
        prev = marker.clone();
        cv::dilate(marker, marker, cv::Mat());    // 膨胀操作
        cv::bitwise_and(marker, inverted, marker); // 约束于原图的反转
    } while (cv::countNonZero(prev != marker) > 0);
    cv::Mat result;
    cv::bitwise_or(src, marker, result);
    return result;
}

cv::Mat contourFill(cv::Mat src) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(src, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    for (int i=0; i<contours.size(); ++i) {
        if (cv::contourArea(contours[i]) < 1000) // 过滤小轮廓
            cv::drawContours(src, contours, i, 255, cv::FILLED);
    }
    return src;
}

cv::Mat connectivityFill(cv::Mat src) {
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(src, labels, stats, centroids);
    for(int i=1; i<n; ++i) {
        if(stats.at<int>(i, cv::CC_STAT_AREA) < 100) // 小区域视为孔洞
            cv::rectangle(src, cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT),
                                      stats.at<int>(i, cv::CC_STAT_TOP),
                                      stats.at<int>(i, cv::CC_STAT_WIDTH),
                                      stats.at<int>(i, cv::CC_STAT_HEIGHT)), 255, -1);
    }
    return src;
}

// 扩展形态学操作处理
cv::Mat ApplyAdvancedOperation(const cv::Mat& src, 
                             const std::string& operation_name) {
    cv::Mat result;
    if (operation_name == "Thinning") {
        result = ThinImage(src);
    } else if (operation_name == "Skeleton") {
        result = Skeletonize(src);
    } else if (operation_name == "FillHoles") {
        result = morphReconstructFill(src);
    } else {
        src.copyTo(result);
    }

    cv::Mat color_result;
    cv::cvtColor(result, color_result, cv::COLOR_GRAY2BGR);
    cv::putText(color_result, operation_name, cv::Point(10, 30),
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    return color_result;
}

}  // namespace

int main() {
  cv::Mat input_image = cv::imread("../assets/src.png", cv::IMREAD_GRAYSCALE);
  if (input_image.empty()) {
    input_image = CreateTestImage();
  }
  std::cout << "input image size" << input_image.size() << std::endl;
  const std::vector<cv::Mat> kernels = CreateStructureElements();
  const std::vector<std::string> kernel_names = {"Rectangle", "Ellipse", "Cross"};  // 矩形、椭圆、十字形

  std::vector<cv::Mat> result_rows;
  for (size_t i = 0; i < kernels.size(); ++i) {
    result_rows.push_back(
        GenerateMorphologyRow(input_image, kernels[i], kernel_names[i]));
  }
  cv::Mat final_output;
  cv::vconcat(result_rows, final_output);

  // 添加高级操作行
  std::vector<cv::Mat> advanced_ops = {
      ApplyAdvancedOperation(input_image, "Thinning"),
      ApplyAdvancedOperation(input_image, "Skeleton"),
      ApplyAdvancedOperation(input_image, "FillHoles")
  };

  cv::Mat advanced_row;
  cv::hconcat(advanced_ops, advanced_row);
  
  cv::imshow("Original Image", input_image);
  cv::imshow("Morphology Operations Comparison", final_output);
  cv::imshow("Extended Operation", advanced_row);
  cv::imwrite("../assets/extended_result.png", advanced_row);
  cv::imwrite("../assets/morphology_result.png", final_output);
  cv::waitKey(0);

  return 0;
}