/**
 *******************************************************************************
 * @file      : hw3.cpp
 * @brief     :
 * @history   :
 *  Version     Date            Author          Note
 *  V0.9.0      yyyy-mm-dd      Immortal-Fates        1. <note>
 *******************************************************************************
 * @attention :
 *******************************************************************************
 *  Copyright (c) Zhejiang University.
 *  All Rights Reserved.
 *******************************************************************************
 */
/* Includes ------------------------------------------------------------------*/
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "ImageStitcher.hpp"

using namespace cv;
using namespace std;
/* Private macro -------------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*
/* Private function prototypes -----------------------------------------------*/
/* External variables --------------------------------------------------------*/
/* Exported function prototypes ----------------------------------------------*/

int main()
{
    vector<string> image_paths;
    // string directory = "../assets/yosemite_test/";
    string directory = "../assets/test_img/";

    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().extension() == ".jpeg") {
            image_paths.push_back(entry.path().string());
            cout << "Found image: " << entry.path().string() << endl;
        }
    }

    // 读入图片时,图片路径的顺序会影响拼接的结果
    sort(image_paths.begin(), image_paths.end());

    vector<Mat> images;
    for (const auto& path : image_paths) {
        Mat image = imread(path);
        if (image.empty()) {
            cerr << "Error: Could not load image " << path << endl;
            return -1;
        }
        images.push_back(image);
    }

    ImageStitcher stitcher;
    Mat stitched_image = stitcher.stitchImages(images);
    imshow("Stitched Image", stitched_image);
    // save img
    string save_path = directory + "stitched_image.jpg";
    imwrite(save_path, stitched_image);

    // press q to exit
    while (waitKey(1) != 'q') {
    }

    return 0;
}