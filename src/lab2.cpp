//
// Created by vuong on 01/12/2021.
//

#include "lab2.h"
namespace DIP {
void DIP::lab2::shift_image(const cv::Mat& img, cv::Mat& new_img) {
  cv::Mat T = (cv::Mat_<double>(2, 3) << 1, 0, 100, 0, 1, 150);
  cv::warpAffine(img, new_img, T, cvSize(img.cols, img.rows));
}
void DIP::lab2::flip_image(const cv::Mat& img, cv::Mat& new_img) {
  cv::Mat T = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, -1, img.rows - 1);
  cv::warpAffine(img, new_img, T, cvSize(img.cols, img.rows));
}
void DIP::lab2::rotate_image(const double angle, const cv::Mat& img,
                             cv::Mat& new_img) {
  cv::Point center = cv::Point(img.cols / 2, img.rows / 2);
  cv::Mat T = cv::getRotationMatrix2D(center, angle, 1);
  cv::warpAffine(img, new_img, T, cvSize(img.cols, img.rows));
}
void lab2::uniform_scale(double scale, const cv::Mat& img, cv::Mat& new_img) {
  cv::Mat T = (cv::Mat_<double>(2, 3) << scale, 0, 0, 0, scale, 0);
  cv::warpAffine(img, new_img, T,
                 cvSize(int(img.cols * scale), int(img.rows * scale)));
}
void lab2::affine2d_image(const cv::Mat& img, cv::Mat& new_img) {
  cv::Point2f srcTri[3];
  srcTri[0] = cv::Point2f(0.f, 0.f);
  srcTri[1] = cv::Point2f(float(img.cols) - 1.f, 0.f);
  srcTri[2] = cv::Point2f(0.f, float(img.rows) - 1.f);
  cv::Point2f dstTri[3];
  dstTri[0] = cv::Point2f(0.f, float(img.rows) * 0.33f);
  dstTri[1] = cv::Point2f(float(img.cols) * 0.85f, float(img.rows) * 0.25f);
  dstTri[2] = cv::Point2f(float(img.cols) * 0.15f, float(img.rows) * 0.7f);
  cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);
  cv::warpAffine(img, new_img, warp_mat, new_img.size());
}
void lab2::bevel_image(double cof, const cv::Mat& img, cv::Mat& new_img) {
  cv::Mat T = (cv::Mat_<double>(2, 3) << 1, cof, 0, 0, 1, 0);
  cv::warpAffine(img, new_img, T,
                 cvSize(int(img.cols + cof * img.rows), img.rows));
}
void lab2::flip_pw(const cv::Mat& img, cv::Mat& new_img) {
  cv::Mat T = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, -1, img.rows - 1);
  cv::Mat temp_right;
  img(cv::Rect(int(new_img.cols / 2), 0, new_img.cols - int(new_img.cols / 2),
               new_img.rows))
      .copyTo(temp_right);
  cv::warpAffine(temp_right, temp_right, T,
                 cvSize(int(temp_right.cols), temp_right.rows));
  img(cv::Rect(0, 0, int(img.cols / 2), img.rows))
      .copyTo(new_img(cv::Rect(0, 0, int(img.cols / 2), img.rows)));

  temp_right.copyTo(new_img(
      cv::Rect(int(new_img.cols / 2), 0, int(new_img.cols / 2), new_img.rows)));
}
void lab2::projective_mapping(const cv::Mat& img, cv::Mat& new_img) {
  int new_w = 480 / 2;
  int new_h = 360 / 2;

  cv::Point2f pts1[4];
  pts1[0] = cv::Point2f(169.f, 55.f);
  pts1[1] = cv::Point2f(364.f, 74.f);
  pts1[2] = cv::Point2f(139.f, 314.f);
  pts1[3] = cv::Point2f(364.f, 314.f);
  cv::Point2f pts2[4];
  pts2[0] = cv::Point2f(0.f, 0.f);
  pts2[1] = cv::Point2f(float(new_w), 0.f);
  pts2[2] = cv::Point2f(0.f, float(new_h));
  pts2[3] = cv::Point2f(float(new_w), float(new_w));
  cv::Mat T_per = cv::getPerspectiveTransform(pts1, pts2);
  cv::Mat temp;
  cv::warpPerspective(img, new_img, T_per, cv::Size(new_w, new_h));
}
void lab2::polynomial_mapping(const Vector6d& A, const Vector6d& B,
                              const cv::Mat& img, cv::Mat& new_img) {
  for (int x = 0; x < img.cols; ++x) {
    for (int y = 0; y < img.rows; ++y) {
      int xnew, ynew;
      xnew = int(std::round(A(0) + x * A(1) + y * A(2)) + x * x * A(3) +
                 x * y * A(4) + y * y * A(5));
      ynew = int(std::round(B(0) + x * B(1) + y * B(2)) + x * x * B(3) +
                 x * y * B(4) + y * y * B(5));
      if (xnew >= 0 && xnew < img.cols && ynew >= 0 && ynew < img.rows)
        new_img.at<cv::Vec3b>(ynew, xnew) = img.at<cv::Vec3b>(y, x);
    }
  }
}
void lab2::sinusoidal_distortion(const cv::Mat& img, cv::Mat& new_img) {
  cv::Mat u = cv::Mat::zeros(img.rows, img.cols, CV_32F);
  cv::Mat v = cv::Mat::zeros(img.rows, img.cols, CV_32F);
  for (int x = 0; x < img.cols; ++x) {
    for (int y = 0; y < img.rows; ++y) {
      u.at<float>(y, x) = float(x + 20 * sin(2 * M_PI * y / 90));
      v.at<float>(y, x) = float(y);
    }
  }
  cv::remap(img, new_img, u, v, cv::INTER_LINEAR);
}
void lab2::undistort_fisheye(const cv::Mat& dist_img, cv::Mat& undist_img) {

  // Camera intrinsic matrix K
  double fx = 2.8498089599609375e+02;
  double fy = 2.8610238647460938e+02;
  double cx = 4.2524438476562500e+02;
  double cy = 3.9846759033203125e+02;
  double k1 = -7.3047108016908169e-03;
  double k2 = 4.3499931693077087e-02;
  double k3 = -4.1283041238784790e-02;
  double k4 = 7.6524601317942142e-03;

  cv::Mat cameraMatrix = cv::Mat(3, 3, cv::DataType<double>::type);
  cv::Mat distortionCoeffs = cv::Mat(4, 1, cv::DataType<double>::type);

  cameraMatrix.at<double>(0, 0) = fx;
  cameraMatrix.at<double>(0, 1) = 0;
  cameraMatrix.at<double>(0, 2) = cx;
  cameraMatrix.at<double>(1, 0) = 0;
  cameraMatrix.at<double>(1, 1) = fy;
  cameraMatrix.at<double>(1, 2) = cy;
  cameraMatrix.at<double>(2, 0) = 0;
  cameraMatrix.at<double>(2, 1) = 0;
  cameraMatrix.at<double>(2, 2) = 1;

  distortionCoeffs.at<double>(0, 0) = k1;
  distortionCoeffs.at<double>(1, 0) = k2;
  distortionCoeffs.at<double>(2, 0) = k3;
  distortionCoeffs.at<double>(3, 0) = k4;

  cv::Mat E = cv::Mat::eye(3, 3, cv::DataType<double>::type);

  cv::Size size = {dist_img.cols, dist_img.rows};

  cv::Mat map1;
  cv::Mat map2;

  cv::fisheye::initUndistortRectifyMap(cameraMatrix, distortionCoeffs, E,
                                       cameraMatrix, size, CV_16SC2, map1,
                                       map2);

  cv::remap(dist_img, undist_img, map1, map2, cv::INTER_LINEAR,
            CV_HAL_BORDER_CONSTANT);
}
bool lab2::pano_stitcher(const std::vector<cv::Mat>& vImgs, cv::Mat& pano) {
  cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);
  cv::Stitcher::Status status = stitcher->stitch(vImgs, pano);
  return status;
}
}  // namespace DIP