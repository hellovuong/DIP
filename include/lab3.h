//
// Created by vuong on 02/12/2021.
//

#ifndef DIP_LAB3_H
#define DIP_LAB3_H
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

#include <Eigen/Core>

#include <ostream>
namespace DIP {
class lab3 {
 public:
  lab3() = default;
  ;
  ~lab3() = default;
  ;

 public:
  // Create noise
  static void impluse_noise(const cv::Mat& im, cv::Mat& noise_im);
  static void speckle_noise(const cv::Mat& im, cv::Mat& noise_im);
  static void gaussian_noise(const double mean, const double stddev,
                             const cv::Mat& im, cv::Mat& noise_im);
  static void poisson_noise(const cv::Mat& im, cv::Mat& noise_im);

  // Low-pass filter
  static void gaussian_filter(cv::Mat& noise_im, cv::Mat& denoise_im);
  static void couterHarmonic_filter(cv::Mat& noise_im, cv::Mat& denoise_im,
                                    int kernel_size, double Q);
  // Non-linear filter
  static void median_filter(cv::Mat& noise_im, cv::Mat& denoise_im, int kSize);
  static void weightedMedian_filter(cv::Mat& noise_im, cv::Mat& denoise_im,
                                    int kSize);
  static void rank_filter(const std::string& method, int kSize,
                          cv::Mat& noise_im, cv::Mat& denoise_im);
  static void adaptive_filter(int minSize, int maxSize, cv::Mat& noise_im,
                              cv::Mat& denoise_im);

  // Edge Detector
  static void roberts_detector(const cv::Mat& im, cv::Mat& dst);
  static void prewitt_detector(const cv::Mat& im, cv::Mat& dst);
  static void sobel_detector(int kernel, const cv::Mat& im, cv::Mat& dst);
  static void laplacian_detector(int kernel_size, const cv::Mat& im,
                                 cv::Mat& dst);
  static void canny_detector(int low_threshold, int kernel_size,
                             const cv::Mat& im, cv::Mat& dst);

 private:
  static uchar adaptive_iterate(const cv::Mat& im, int row, int col,
                                int kernelSize, int maxSize);
};
}  // namespace DIP
#endif  // DIP_LAB3_H
