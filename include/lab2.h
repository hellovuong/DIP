//
// Created by vuong on 01/12/2021.
//

#ifndef DIP_LAB2_H
#define DIP_LAB2_H

#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/stitching.hpp"

#include <Eigen/Core>

#include <ostream>

namespace DIP {
typedef Eigen::Matrix<double, 6, 1> Vector6d;
class lab2 {
 public:
  lab2() = default;
  ~lab2() = default;
  static void shift_image(const cv::Mat& img, cv::Mat& new_img);
  static void flip_image(const cv::Mat& img, cv::Mat& new_img);
  static void rotate_image(double angle, const cv::Mat& img, cv::Mat& new_img);
  static void uniform_scale(double scale, const cv::Mat& img, cv::Mat& new_img);
  static void affine2d_image(const cv::Mat& img, cv::Mat& new_img);
  static void bevel_image(double scale, const cv::Mat& img, cv::Mat& new_img);
  static void flip_pw(const cv::Mat& img, cv::Mat& new_img);
  static void projective_mapping(const cv::Mat& img, cv::Mat& new_img);
  static void polynomial_mapping(const Vector6d& A, const Vector6d& B,
                                 const cv::Mat& img, cv::Mat& new_img);
  static void sinusoidal_distortion(const cv::Mat& img, cv::Mat& new_img);
  static void undistort_fisheye(const cv::Mat& dist_img, cv::Mat& undist_img);
  static bool pano_stitcher(const std::vector<cv::Mat>& vImgs, cv::Mat& pano);
};
}  // namespace DIP
#endif  // DIP_LAB2_H
