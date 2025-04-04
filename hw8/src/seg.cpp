#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main() {
    Mat src = imread("../assets/imgC.png");
    if (src.empty()) {
        cout << "图像加载失败" << endl;
        return -1;
    }
  
    // ================= 预处理阶段 =================
    Mat blurred;
    GaussianBlur(src, blurred, Size(3,3), 1);
  
    // 转换为灰度并二值化
    Mat gray, thresh;
    cvtColor(blurred, gray, COLOR_BGR2GRAY);
    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
    imshow("1. 二值化图像", thresh);
    imwrite("../assets/result/1_after_binarization.jpg", thresh);
  
    // 形态学操作
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3,3));
    morphologyEx(thresh, thresh, MORPH_OPEN, kernel, Point(-1,-1), 2);
    morphologyEx(thresh, thresh, MORPH_CLOSE, kernel, Point(-1,-1), 2);
    imshow("2. 形态学处理", thresh);
    imwrite("../assets/result/2_after_morphology.jpg", thresh);

    // ================= 距离变换阶段 =================
    Mat dist;
    distanceTransform(thresh, dist, DIST_L2, 5);
  
    // 归一化距离变换结果用于显示
    Mat dist_display;
    normalize(dist, dist_display, 0, 255, NORM_MINMAX);
    dist_display.convertTo(dist_display, CV_8U);
    applyColorMap(dist_display, dist_display, COLORMAP_JET); // 应用伪彩色
    imshow("3. 距离变换", dist_display);
    imwrite("../assets/result/3_distance_transform.jpg", dist_display);

    // ================= 标记生成阶段 =================
    double minVal, maxVal;
    minMaxLoc(dist, &minVal, &maxVal);
  
    // 生成前景标记
    Mat fg_mask;
    threshold(dist, fg_mask, 0.5*maxVal, 255, THRESH_BINARY);
    fg_mask.convertTo(fg_mask, CV_8U);
  
    // 显示前景标记
    Mat fg_display;
    cvtColor(fg_mask, fg_display, COLOR_GRAY2BGR);
    fg_display.setTo(Scalar(0,255,0), fg_mask); // 绿色前景
    imshow("4. 前景标记", fg_display);
    imwrite("../assets/result/4_foreground_mask.jpg", fg_display);

    // 生成背景标记
    Mat bg_mask;
    dilate(thresh, bg_mask, kernel, Point(-1,-1), 1);
    bg_mask = bg_mask - fg_mask;
  
    // 显示背景标记
    Mat bg_display;
    cvtColor(bg_mask, bg_display, COLOR_GRAY2BGR);
    bg_display.setTo(Scalar(0,0,255), bg_mask); // 红色背景
    imshow("5. 背景标记", bg_display);
    imwrite("../assets/result/5_background_mask.jpg", bg_display);

    // ================= 标记合成阶段 =================
    Mat markers(thresh.size(), CV_32S, Scalar(0));
    markers.setTo(1, bg_mask/255);   // 背景=1 (蓝色)
    markers.setTo(2, fg_mask/255);    // 前景=2 (绿色)
    markers.setTo(0, (bg_mask == 0) & (fg_mask == 0)); // 未知区域=0 (红色)

    // 可视化标记图
    Mat markers_display = Mat::zeros(markers.size(), CV_8UC3);
    markers_display.setTo(Scalar(255,0,0), markers == 1);  // 背景-蓝色
    markers_display.setTo(Scalar(0,255,0), markers == 2);  // 前景-绿色
    markers_display.setTo(Scalar(0,0,255), markers == 0); // 未知-红色
    imshow("6. 合成标记图", markers_display);
    imwrite("../assets/result/6_combined_markers.jpg", markers_display);

    // ================= GrabCut分割 =================
    
    Mat grab_mask(bg_mask.size(), CV_8UC1, GC_PR_BGD); // 初始化可能背景
    grab_mask.setTo(GC_FGD, bg_mask);    // 确定前景
    grab_mask.setTo(GC_BGD, fg_mask);    // 确定背景

    Mat bgdModel, fgdModel;
    grabCut(src, grab_mask, Rect(), bgdModel, fgdModel, 3, GC_INIT_WITH_MASK);

    // 生成最终掩膜
    Mat result_mask;
    compare(grab_mask, GC_FGD, result_mask, CMP_EQ);
    // 创建3通道掩膜用于乘法
    Mat result_mask_3ch;
    cvtColor(result_mask, result_mask_3ch, COLOR_GRAY2BGR);

    // 与原图合成（按位与操作）
    Mat result;
    bitwise_and(src, result_mask_3ch, result);
    
    imshow("7. 最终分割结果", result);
    imwrite("../assets/result/7_final_result.jpg", result);

    // ================= 窗口布局优化 =================
    // 调整窗口位置避免重叠
    moveWindow("1. 二值化图像", 50, 50);
    moveWindow("2. 形态学处理", 450, 50);
    moveWindow("3. 距离变换", 850, 50);
    moveWindow("4. 前景标记", 50, 400);
    moveWindow("5. 背景标记", 450, 400);
    moveWindow("6. 合成标记图", 850, 400);
    moveWindow("7. 最终分割结果", 1250, 200);

    waitKey(0);
    return 0;
}