//
// Created by vuong on 30/11/2021.
//

#ifndef DIP_LAB1_H
#define DIP_LAB1_H
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ostream>
#include "../thirdparty/matplotlib-cpp/matplotlibcpp.h"

namespace DIP {
class lab1 {
 public:
  lab1() = default;
  ~lab1() = default;

 public:
  void histogram(cv::Mat& img, std::vector<cv::Mat>& res,
                 std::vector<cv::Mat>& cum_hist);
  void plot_hist(cv::Mat& imHist, const std::vector<cv::Mat>& vHist) const;
  void drs(cv::Mat& img, std::vector<cv::Mat>& vHist, const double alpha);
  void uniform_trans(cv::Mat& img, std::vector<cv::Mat>& vHist);
  void exp_trans(cv::Mat& img, std::vector<cv::Mat>& vHist, const double alpha);
  void ray_trans(cv::Mat& img, std::vector<cv::Mat>& vHist, const double alpha);
  void trans23(cv::Mat& img, std::vector<cv::Mat>& vHist);
  void hyper_trans(cv::Mat& img, std::vector<cv::Mat>& vHist,
                   const double alpha);
  static void profile(const cv::Mat& img, cv::Mat& pro_x, cv::Mat& pro_y);
  static void projection(const cv::Mat& img, cv::Mat& proj_x, cv::Mat& proj_y);

 private:
  const int histSize = 256;
  const int hist_w = 512, hist_h = 400;
};
}  // namespace DIP
#endif  // DIP_LAB1_H
