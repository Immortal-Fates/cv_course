#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class ImageStitcher
{
   public:
    ImageStitcher()
    {
        sift = SIFT::create();
    }

    vector<KeyPoint> detectAndComputeFeatures(const Mat& image)
    {
        Mat gray;
        cvtColor(image, gray, COLOR_BGR2GRAY);
        vector<KeyPoint> keypoints;
        Mat descriptors;
        sift->detectAndCompute(gray, noArray(), keypoints, descriptors);
        descriptors_list.push_back(descriptors);
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
            src_pts.push_back(keypoints1[m.queryIdx].pt);
            dst_pts.push_back(keypoints2[m.trainIdx].pt);
        }
        Mat M;
        findHomography(src_pts, dst_pts, M, RANSAC, 5.0);
        return M;
    }

    Mat warpImage(const Mat& image, const Mat& M, int width, int height)
    {
        Mat warped_image;
        warpPerspective(image, warped_image, M, Size(width, height));
        return warped_image;
    }

    Mat blendImages(const Mat& image1, const Mat& image2, int overlap_width)
    {
        Mat blended_image = Mat::zeros(image1.size(), image1.type());
        for (int i = 0; i < image1.rows; ++i) {
            for (int j = 0; j < image1.cols; ++j) {
                double alpha = min(1.0, max(0.0, (double)(overlap_width - j) / overlap_width));
                blended_image.at<Vec3b>(i, j) = alpha * image1.at<Vec3b>(i, j) + (1 - alpha) * image2.at<Vec3b>(i, j);
            }
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

int main()
{
    vector<string> image_paths = {"../assets/yosemite_test/yosemite1.jpg", "../assets/yosemite_test/yosemite2.jpg", "../assets/yosemite_test/yosemite3.jpg", "../assets/yosemite_test/yosemite4.jpg"};
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
    waitKey(0);

    return 0;
}