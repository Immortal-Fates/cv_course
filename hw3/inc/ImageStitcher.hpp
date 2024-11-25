/**
 *******************************************************************************
 * @file      : ImageStitcher.hpp
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
/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef IMAGESTITCHER_HPP_
#define IMAGESTITCHER_HPP_

/* Includes ------------------------------------------------------------------*/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;
/* Exported macro ------------------------------------------------------------*/
/* Exported constants --------------------------------------------------------*/
/* Exported types ------------------------------------------------------------*/
class ImageStitcher
{
   public:
    ImageStitcher()
    {
        sift = SIFT::create();
    }
    ~ImageStitcher() = default;

    vector<KeyPoint> detectAndComputeFeatures(const Mat& image)
    {
        Mat gray;
        cvtColor(image, gray, COLOR_BGR2GRAY);
        vector<KeyPoint> keypoints;
        Mat descriptors;
        sift->detectAndCompute(gray, noArray(), keypoints, descriptors);
        descriptors_list.push_back(descriptors);

        // 将图像中的关键点绘制到图像上
        Mat image_with_keypoints;
        drawKeypoints(image, keypoints, image_with_keypoints, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        imshow("Image with Keypoints", image_with_keypoints);
        waitKey(0);
        return keypoints;
    }

    vector<DMatch> matchFeatures(const Mat& descriptors1, const Mat& descriptors2)
    {
        BFMatcher bf(NORM_L2);
        vector<vector<DMatch>> matches;
        bf.knnMatch(descriptors1, descriptors2, matches, 2);
        vector<DMatch> good_matches;
        for (const auto& m : matches) {
            if (m[0].distance < 0.75 * m[1].distance) {
                good_matches.push_back(m[0]);
            }
        }
        // 将匹配的特征点绘制到图像上
        Mat image_matches;
        drawMatches(images[0], keypoints_list[0], images[1], keypoints_list[1], good_matches, image_matches);
        imshow("Image Matches", image_matches);
        waitKey(0);

        return good_matches;
    }

    /**
     * @brief       计算单应性矩阵
     * @arg         None
     * @retval       None
     * @note        None
     */
    Mat estimateHomography(const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2, const vector<DMatch>& good_matches)
    {
        vector<Point2f> src_pts, dst_pts;
        for (const auto& m : good_matches) {
            src_pts.push_back(keypoints1[m.queryIdx].pt);
            dst_pts.push_back(keypoints2[m.trainIdx].pt);
        }
        Mat M;
        findHomography(src_pts, dst_pts, M, RANSAC, 5.0);
        return M;
    }

    /**
     * @brief       图像进行透视变换
     * @param        image:
     * @param        M:
     * @param        width:
     * @param        height:
     * @arg         None
     * @retval       None
     * @note        None
     */
    Mat warpImage(const Mat& image, const Mat& M, int width, int height)
    {
        Mat warped_image;
        warpPerspective(image, warped_image, M, Size(width, height));
        return warped_image;
    }

    Mat blendImages(const Mat& image1, const Mat& image2, int overlap_width)
    {
        vector<Mat> pyr1, pyr2, pyr_blend;

        // 构建高斯金字塔
        Mat temp1 = image1.clone();
        Mat temp2 = image2.clone();
        for (int i = 0; i < levels; ++i) {
            pyr1.push_back(temp1);
            pyr2.push_back(temp2);
            pyrDown(temp1, temp1);
            pyrDown(temp2, temp2);
        }

        // 构建拉普拉斯金字塔
        for (int i = 0; i < levels - 1; ++i) {
            Mat lap1, lap2;
            pyrUp(pyr1[i + 1], lap1, pyr1[i].size());
            pyrUp(pyr2[i + 1], lap2, pyr2[i].size());
            lap1 = pyr1[i] - lap1;
            lap2 = pyr2[i] - lap2;
            pyr_blend.push_back(lap1 + lap2);
        }

        // 重建图像
        Mat blended_image = pyr_blend.back();
        for (int i = levels - 2; i >= 0; --i) {
            pyrUp(blended_image, blended_image, pyr_blend[i].size());
            blended_image += pyr_blend[i];
        }

        return blended_image;
    }

    Mat stitchImages(const vector<Mat>& images)
    {
        vector<vector<KeyPoint>> keypoints_list;
        for (const auto& image : images) {
            keypoints_list.push_back(detectAndComputeFeatures(image));
        }

        vector<Mat> homographies;
        for (size_t i = 0; i < images.size() - 1; ++i) {
            vector<DMatch> good_matches = matchFeatures(descriptors_list[i], descriptors_list[i + 1]);
            Mat M = estimateHomography(keypoints_list[i], keypoints_list[i + 1], good_matches);
            homographies.push_back(M);
        }

        int width = images[0].cols + images.back().cols;
        int height = max(images[0].rows, images.back().rows);

        vector<Mat> warped_images;
        warped_images.push_back(images[0]);
        for (size_t i = 0; i < images.size() - 1; ++i) {
            Mat warped_image = warpImage(images[i + 1], homographies[i], width, height);
            warped_images.push_back(warped_image);
        }

        Mat stitched_image = warped_images[0];
        for (size_t i = 0; i < warped_images.size() - 1; ++i) {
            int overlap_width = min(warped_images[i].cols, warped_images[i + 1].cols);
            Mat blended_image = blendImages(warped_images[i], warped_images[i + 1], overlap_width);
            stitched_image = blended_image;
        }

        return stitched_image;
    }

   private:
    Ptr<SIFT> sift;
    vector<Mat> descriptors_list;
};
/* Exported variables --------------------------------------------------------*/
/* Exported function prototypes ----------------------------------------------*/

#endif /* IMAGESTITCHER_HPP_ */
