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
#include <math.h>

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
        return good_matches;
    }

    Mat estimateHomography(const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2, const vector<DMatch>& good_matches)
    {
        vector<Point2f> src_pts, dst_pts;
        for (const auto& m : good_matches) {
            dst_pts.push_back(keypoints1[m.queryIdx].pt);
            src_pts.push_back(keypoints2[m.trainIdx].pt);
        }
        // 检查匹配点的数量是否足够
        if (src_pts.size() < 4 || dst_pts.size() < 4) {
            throw std::runtime_error("Not enough matches to compute homography");
        }

        Mat M = findHomography(src_pts, dst_pts, RANSAC, 5.0);

        // 检查返回的单应矩阵是否为 3x3
        if (M.rows != 3 || M.cols != 3) {
            throw std::runtime_error("Homography matrix is not 3x3");
        }
        Mat M_float;
        M.convertTo(M_float, CV_32F);
        cout << "Homography matrix" << M_float << endl;
        return M_float;
    }

    // Function to create a Gaussian pyramid
    void createGaussianPyramid(const Mat& image, vector<Mat>& gaussianPyramid, int levels)
    {
        gaussianPyramid.push_back(image.clone());
        Mat currentImage = image.clone();
        for (int i = 1; i < levels; ++i) {
            if (currentImage.cols % 2 != 0) {
                resize(currentImage, currentImage, Size(currentImage.cols - 1, currentImage.rows));
            }
            if (currentImage.rows % 2 != 0) {
                resize(currentImage, currentImage, Size(currentImage.cols, currentImage.rows - 1));
            }
            Mat downsampled;
            pyrDown(currentImage, downsampled);
            gaussianPyramid.push_back(downsampled);
            currentImage = downsampled;
        }
    }

    // Function to create a Laplacian pyramid
    void createLaplacianPyramid(const vector<Mat>& gaussianPyramid, vector<Mat>& laplacianPyramid, int levels)
    {
        for (int i = 0; i < levels - 1; ++i) {
            Mat gaussianNextLevel;
            // cout << "gaussianPyramid[i].size():" << gaussianPyramid[i].size() << endl;
            // cout << "gaussianPyramid[i + 1].size():" << gaussianPyramid[i + 1].size() << endl;

            pyrUp(gaussianPyramid[i + 1], gaussianNextLevel);
            Mat laplacian = gaussianPyramid[i] - gaussianNextLevel;
            laplacianPyramid.push_back(laplacian);
        }
        laplacianPyramid.push_back(gaussianPyramid[levels - 1]);
    }
    Mat blendImages(Mat& image1, Mat& image2)
    {
        int levels = 4;  // Number of levels in the pyramid
        // 将图像resize为2^4=16的倍数
        if (image1.cols % 16 != 0) {
            resize(image1, image1, Size(image1.cols - image1.cols % 16, image1.rows));
        }
        if (image1.rows % 16 != 0) {
            resize(image1, image1, Size(image1.cols, image1.rows - image1.rows % 16));
        }
        resize(image2, image2, Size(image1.cols, image1.rows));

        // Create Gaussian pyramids for both images
        vector<Mat> gaussianPyramid1, gaussianPyramid2;
        createGaussianPyramid(image1, gaussianPyramid1, levels);
        createGaussianPyramid(image2, gaussianPyramid2, levels);

        // Create Laplacian pyramids for both images
        vector<Mat> laplacianPyramid1, laplacianPyramid2;
        createLaplacianPyramid(gaussianPyramid1, laplacianPyramid1, levels);
        createLaplacianPyramid(gaussianPyramid2, laplacianPyramid2, levels);

        static int count = 0;
        count++;
        cout << "count: " << count << endl;

        // Blend the images at each level of the pyramid
        vector<Mat> blendedPyramid;
        for (int i = 0; i < levels; ++i) {
            int width = laplacianPyramid1[i].cols;
            int height = laplacianPyramid1[i].rows;
            Mat blended(height, width, laplacianPyramid1[i].type());
            cout << "width: " << width << " height: " << height << endl;
            Mat mask = Mat::zeros(height, width, CV_32F);
            float begin_col = -1.0f;
            int overlap_width = 0;
            // 转换为灰度图再进行计算
            Mat gray1, gray2;
            cvtColor(laplacianPyramid1[i], gray1, COLOR_BGR2GRAY);
            cvtColor(laplacianPyramid2[i], gray2, COLOR_BGR2GRAY);
            for (int col = 0; col < width; ++col) {
                if (sum(gray1.col(col))[0] > 0 && sum(gray2.col(col))[0] > 0) {
                    if (begin_col == -1.0f) {
                        begin_col = col;
                    }
                    overlap_width++;
                } else if (sum(gray1.col(col))[0] > 0) {
                    mask.col(col) = 1;
                } else if (sum(gray2.col(col))[0] > 0) {
                    mask.col(col) = 0;
                }
            }
            // 重叠部分
            for (int col = begin_col; col < begin_col + overlap_width; ++col) {
                float alpha = 1 - (col - begin_col) / overlap_width;
                mask.col(col) = alpha;
            }
            Mat mask2 = 1 - mask;
            Mat expand_mask1, expand_mask2;
            vector<Mat> channels1(3, mask), channels2(3, mask2);
            merge(channels1, expand_mask1);
            merge(channels2, expand_mask2);

            Mat leftPart, rightPart;
            laplacianPyramid1[i].convertTo(leftPart, CV_32F);
            laplacianPyramid2[i].convertTo(rightPart, CV_32F);
            // 对于重叠部分, 使用简单的平均值进行混合,其他部分使用原来有的图像
            blended = leftPart.mul(expand_mask1) + rightPart.mul(expand_mask2);
            blendedPyramid.push_back(blended);
        }

        // Reconstruct
        Mat blendedImage = blendedPyramid[levels - 1];
        for (int i = levels - 2; i >= 0; --i) {
            pyrUp(blendedImage, blendedImage, blendedPyramid[i].size());
            blendedImage += blendedPyramid[i];
        }

        blendedImage.convertTo(blendedImage, CV_8UC3);
        // 去除后面纯黑的列
        int col = blendedImage.cols - 1;
        Mat gray_img;
        cvtColor(blendedImage, gray_img, COLOR_BGR2GRAY);
        while (sum(gray_img.col(col))[0] == 0) {
            col--;
        }
        blendedImage = blendedImage.colRange(0, col + 1);

        return blendedImage;
    }

    Mat stitchTwoImages(Mat& img1, Mat& img2, Mat& aff_mat)
    {
        // 构建一个空的全景图像来保存拼接结果
        int height = std::max(img1.rows, img2.rows);
        int width = img1.cols + img2.cols;
        Mat pano(height, width, CV_8UC3, Scalar(0, 0, 0));

        // 将仿射矩阵传递给 warpAffine
        aff_mat.convertTo(aff_mat, CV_64FC1);
        Mat warp_mat(2, 3, CV_64FC1);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                warp_mat.at<double>(i, j) = aff_mat.at<double>(i, j);
            }
        }

        Mat src1 = Mat::zeros(height, width, img1.type());
        img1.copyTo(src1.rowRange(0, img1.rows).colRange(0, img1.cols));
        Mat src2 = Mat::zeros(height, width, img2.type());
        warpAffine(img2, src2, warp_mat, src2.size());

        // 混合两张图像: pyramid blend
        pano = blendImages(src1, src2);

        return pano;
    }

    Mat stitchImages(const vector<Mat>& images)
    {
        if (images.size() < 2) {
            throw std::runtime_error("Need at least two images to stitch");
        }
        Mat pano = images[0].clone();
        for (int i = 1; i < images.size(); i++) {
            vector<KeyPoint> keypoints1 = detectAndComputeFeatures(pano);
            vector<KeyPoint> keypoints2 = detectAndComputeFeatures(images[i]);

            vector<DMatch> good_matches = matchFeatures(getDescriptors(0), getDescriptors(1));
            Mat homography = estimateHomography(keypoints1, keypoints2, good_matches);
            Mat temp_image2 = images[i].clone();
            Mat temp = stitchTwoImages(pano, temp_image2, homography);
            pano = temp.clone();
            // 清空描述符列表
            descriptors_list.clear();
        }

        imshow("Stitched Image", pano);
        waitKey(0);
        return pano;
    }

    Mat getDescriptors(int index) const
    {
        return descriptors_list[index];
    }

   private:
    Ptr<SIFT> sift;
    vector<Mat> descriptors_list;
};

/* Exported variables --------------------------------------------------------*/
/* Exported function prototypes ----------------------------------------------*/

#endif /* IMAGESTITCHER_HPP_ */
