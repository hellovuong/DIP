#include "lab1.h"

#include <iostream>

namespace DIP {
void lab1::histogram(cv::Mat& orig_img, std::vector<cv::Mat>& vHist,
                     std::vector<cv::Mat>& vCumHist) {
  cv::Mat img = orig_img.clone();

  std::vector<cv::Mat> bgr_planes;
  split(orig_img, bgr_planes);

  float range[] = {0, 256};  // the upper boundary is exclusive
  const float* histRange[] = {range};

  cv::Mat b_hist, g_hist, r_hist;
  cv::Mat b_hist_norm, g_hist_norm, r_hist_norm;
  cv::calcHist(&bgr_planes[0], 1, nullptr, cv::Mat(), b_hist, 1, &histSize,
               histRange);
  cv::calcHist(&bgr_planes[1], 1, nullptr, cv::Mat(), g_hist, 1, &histSize,
               histRange);
  cv::calcHist(&bgr_planes[2], 1, nullptr, cv::Mat(), r_hist, 1, &histSize,
               histRange);

  vHist[0] = b_hist;
  vHist[1] = g_hist;
  vHist[2] = r_hist;

  // channel 0
  vCumHist[0] = b_hist.clone();
  for (int i = 1; i < b_hist.rows; ++i) {
    vCumHist[0].at<float>(i) += vCumHist[0].at<float>(i - 1);
  }
  vCumHist[0] /= orig_img.rows * orig_img.cols;

  // channel 1
  vCumHist[1] = g_hist.clone();
  for (int i = 1; i < g_hist.rows; ++i) {
    vCumHist[1].at<float>(i) += vCumHist[1].at<float>(i - 1);
  }
  vCumHist[1] /= orig_img.rows * orig_img.cols;

  // channel 2
  vCumHist[2] = r_hist.clone();
  for (int i = 1; i < r_hist.rows; ++i) {
    vCumHist[2].at<float>(i) += vCumHist[2].at<float>(i - 1);
  }
  vCumHist[2] /= orig_img.rows * orig_img.cols;
}
void lab1::drs(cv::Mat& img, std::vector<cv::Mat>& vHist, const double alpha) {
  std::vector<cv::Mat> BGR_Image;
  double min[3], max[3];
  int rows = img.rows;
  int cols = img.cols;

  // Convert Each Channel Pixel Values from 8U to 32F
  // Copy Image.data values to RGB_Image for Extracting All Three Frames [R, G,
  // B]
  img.convertTo(img, CV_32FC3);
  split(img, BGR_Image);

  // Store minimum and maximum values of [R, G, B] channels separately.
  for (int i = 0; i < 3; i++) {
    minMaxLoc(BGR_Image[i], &min[i], &max[i]);
    BGR_Image[i].convertTo(BGR_Image[i], CV_8U);
  }
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      float B = BGR_Image[0].at<uchar>(row, col) - min[0];
      float G = BGR_Image[1].at<uchar>(row, col) - min[1];
      float R = BGR_Image[2].at<uchar>(row, col) - min[2];

      double contrast_B = max[0] - min[0];
      double contrast_G = max[1] - min[1];
      double contrast_R = max[2] - min[2];

      BGR_Image[0].at<uchar>(row, col) =
          std::pow(B, alpha) * 255 / std::pow(contrast_B, alpha);
      BGR_Image[1].at<uchar>(row, col) =
          std::pow(G, alpha) * 255 / std::pow(contrast_G, alpha);
      BGR_Image[2].at<uchar>(row, col) =
          std::pow(R, alpha) * 255 / std::pow(contrast_R, alpha);
    }
  }

  merge(BGR_Image, img);
  std::vector<cv::Mat> vCumHist;
  vCumHist.resize(3);
  histogram(img, vHist, vCumHist);
}
void lab1::uniform_trans(cv::Mat& img, std::vector<cv::Mat>& vHist) {
  std::vector<cv::Mat> vCumHist, BGR_Image;
  double min[3], max[3];

  vCumHist.resize(3);
  histogram(img, vHist, vCumHist);

  // Convert Each Channel Pixel Values from 8U to 32F
  // Copy Image.data values to RGB_Image for Extracting All Three Frames [R, G,
  // B]
  img.convertTo(img, CV_32F);
  split(img, BGR_Image);

  // Store minimum and maximum values of [R, G, B] channels separately.
  for (int i = 0; i < 3; i++) {
    minMaxLoc(BGR_Image[i], &min[i], &max[i]);
    BGR_Image[i].convertTo(BGR_Image[i], CV_8U);
  }

  for (int row = 0; row < img.rows; row++) {
    for (int col = 0; col < img.cols; col++) {
      double contrast_B = max[0] - min[0];
      double contrast_G = max[1] - min[1];
      double contrast_R = max[2] - min[2];

      BGR_Image[0].at<uchar>(row, col) =
          contrast_B * vCumHist[0].at<float>(BGR_Image[0].at<uchar>(row, col)) +
          min[0];
      BGR_Image[1].at<uchar>(row, col) =
          contrast_G * vCumHist[1].at<float>(BGR_Image[1].at<uchar>(row, col)) +
          min[1];
      BGR_Image[2].at<uchar>(row, col) =
          contrast_R * vCumHist[2].at<float>(BGR_Image[2].at<uchar>(row, col)) +
          min[2];
    }
  }

  merge(BGR_Image, img);

  for (auto& i : vHist) i.setTo(cv::Scalar::all(0));
  for (auto& i : vCumHist) i.setTo(cv::Scalar::all(0));

  histogram(img, vHist, vCumHist);
}
void lab1::plot_hist(cv::Mat& imHist, const std::vector<cv::Mat>& vHist) const {
  const cv::Mat b_hist = vHist[0];
  const cv::Mat g_hist = vHist[1];
  const cv::Mat r_hist = vHist[2];

  cv::Mat b_hist_norm, r_hist_norm, g_hist_norm;

  cv::normalize(b_hist, b_hist_norm, 0, imHist.rows, cv::NORM_MINMAX, -1,
                cv::Mat());
  cv::normalize(g_hist, g_hist_norm, 0, imHist.rows, cv::NORM_MINMAX, -1,
                cv::Mat());
  cv::normalize(r_hist, r_hist_norm, 0, imHist.rows, cv::NORM_MINMAX, -1,
                cv::Mat());

  for (int i = 1; i < histSize; i++) {
    cv::rectangle(imHist,
                  cv::Point(2 * i, imHist.rows - b_hist_norm.at<float>(i)),
                  cv::Point(2 * (i - 1), imHist.rows), cv::Scalar(255, 0, 0));
    cv::rectangle(imHist,
                  cv::Point(2 * i, imHist.rows - g_hist_norm.at<float>(i)),
                  cv::Point(2 * (i - 1), imHist.rows), cv::Scalar(0, 255, 0));
    cv::rectangle(imHist,
                  cv::Point(2 * i, imHist.rows - r_hist_norm.at<float>(i)),
                  cv::Point(2 * (i - 1), imHist.rows), cv::Scalar(0, 0, 255));
  }
}
void lab1::exp_trans(cv::Mat& img, std::vector<cv::Mat>& vHist,
                     const double alpha) {
  std::vector<cv::Mat> vCumHist, BGR_Image;
  double min[3], max[3];

  vCumHist.resize(3);
  histogram(img, vHist, vCumHist);

  // Convert Each Channel Pixel Values from 8U to 32F
  // Copy Image.data values to RGB_Image for Extracting All Three Frames [R, G,
  // B]
  img.convertTo(img, CV_32FC3);
  split(img, BGR_Image);

  // Store minimum and maximum values of [R, G, B] channels separately.
  for (int i = 0; i < 3; i++) {
    minMaxLoc(BGR_Image[i], &min[i], &max[i]);
    BGR_Image[i].convertTo(BGR_Image[i], CV_8U);
  }

  for (int row = 0; row < img.rows; row++) {
    for (int col = 0; col < img.cols; col++) {
      float B = vCumHist[0].at<float>(BGR_Image[0].at<uchar>(row, col));
      float G = vCumHist[1].at<float>(BGR_Image[1].at<uchar>(row, col));
      float R = vCumHist[2].at<float>(BGR_Image[2].at<uchar>(row, col));

      BGR_Image[0].at<uchar>(row, col) = cv::saturate_cast<uchar>(
          255 * (min[0] - 1 / alpha * std::log(1 - B)));
      BGR_Image[1].at<uchar>(row, col) = cv::saturate_cast<uchar>(
          255 * (min[1] - 1 / alpha * std::log(1 - G)));
      BGR_Image[2].at<uchar>(row, col) = cv::saturate_cast<uchar>(
          255 * (min[2] - 1 / alpha * std::log(1 - R)));
    }
  }
  merge(BGR_Image, img);

  for (auto& i : vHist) i.setTo(cv::Scalar::all(0));
  for (auto& i : vCumHist) i.setTo(cv::Scalar::all(0));

  histogram(img, vHist, vCumHist);
}
void lab1::ray_trans(cv::Mat& img, std::vector<cv::Mat>& vHist,
                     const double alpha) {
  std::vector<cv::Mat> vCumHist, BGR_Image;
  double min[3], max[3];
  double alpha2 = 2 * alpha * alpha;
  vCumHist.resize(3);
  histogram(img, vHist, vCumHist);

  // Convert Each Channel Pixel Values from 8UC3 to 32FC3
  // Copy Image.data values to RGB_Image for Extracting All Three Frames [R, G,
  // B]
  img.convertTo(img, CV_32FC3);
  split(img, BGR_Image);

  // Store minimum and maximum values of [R, G, B] channels separately.
  for (int i = 0; i < 3; i++) {
    minMaxLoc(BGR_Image[i], &min[i], &max[i]);
    BGR_Image[i].convertTo(BGR_Image[i], CV_8U);
  }
  for (int row = 0; row < img.rows; row++) {
    for (int col = 0; col < img.cols; col++) {
      float B = vCumHist[0].at<float>(BGR_Image[0].at<uchar>(row, col));
      float G = vCumHist[1].at<float>(BGR_Image[1].at<uchar>(row, col));
      float R = vCumHist[2].at<float>(BGR_Image[2].at<uchar>(row, col));

      BGR_Image[0].at<uchar>(row, col) = cv::saturate_cast<uchar>(
          255 * (min[0] + std::sqrt(alpha2 * std::log(1 / (1 - B)))));
      BGR_Image[1].at<uchar>(row, col) = cv::saturate_cast<uchar>(
          255 * (min[1] + std::sqrt(alpha2 * std::log(1 / (1 - G)))));
      BGR_Image[2].at<uchar>(row, col) = cv::saturate_cast<uchar>(
          255 * (min[2] + std::sqrt(alpha2 * std::log(1 / (1 - R)))));
    }
  }
  merge(BGR_Image, img);

  for (auto& i : vHist) i.setTo(cv::Scalar::all(0));
  for (auto& i : vCumHist) i.setTo(cv::Scalar::all(0));

  histogram(img, vHist, vCumHist);
}
void lab1::trans23(cv::Mat& img, std::vector<cv::Mat>& vHist) {
  std::vector<cv::Mat> vCumHist, BGR_Image;
  vCumHist.resize(3);
  histogram(img, vHist, vCumHist);

  // Convert Each Channel Pixel Values from 8U to 32F
  // Copy Image.data values to RGB_Image for Extracting All Three Frames [R, G,
  // B]
  img.convertTo(img, CV_32FC3);
  split(img, BGR_Image);

  // Store minimum and maximum values of [R, G, B] channels separately.
  for (int i = 0; i < 3; i++) {
    BGR_Image[i].convertTo(BGR_Image[i], CV_8U);
  }

  for (int row = 0; row < img.rows; row++) {
    for (int col = 0; col < img.cols; col++) {
      float B = vCumHist[0].at<float>(BGR_Image[0].at<uchar>(row, col));
      float G = vCumHist[1].at<float>(BGR_Image[1].at<uchar>(row, col));
      float R = vCumHist[2].at<float>(BGR_Image[2].at<uchar>(row, col));

      BGR_Image[0].at<uchar>(row, col) =
          cv::saturate_cast<uchar>(255 * powf(B, (float)2 / 3));
      BGR_Image[1].at<uchar>(row, col) =
          cv::saturate_cast<uchar>(255 * powf(G, (float)2 / 3));
      BGR_Image[2].at<uchar>(row, col) =
          cv::saturate_cast<uchar>(255 * powf(R, (float)2 / 3));
    }
  }
  merge(BGR_Image, img);

  for (auto& i : vHist) i.setTo(cv::Scalar::all(0));
  for (auto& i : vCumHist) i.setTo(cv::Scalar::all(0));

  histogram(img, vHist, vCumHist);
}
void lab1::hyper_trans(cv::Mat& img, std::vector<cv::Mat>& vHist,
                       const double alpha) {
  std::vector<cv::Mat> vCumHist, BGR_Image;
  vCumHist.resize(3);
  histogram(img, vHist, vCumHist);

  // Convert Each Channel Pixel Values from 8U to 32F
  // Copy Image.data values to RGB_Image for Extracting All Three Frames [R, G,
  // B]
  img.convertTo(img, CV_32FC3);
  split(img, BGR_Image);

  // Store minimum and maximum values of [R, G, B] channels separately.
  for (int i = 0; i < 3; i++) {
    BGR_Image[i].convertTo(BGR_Image[i], CV_8U);
  }

  for (int row = 0; row < img.rows; row++) {
    for (int col = 0; col < img.cols; col++) {
      float B = vCumHist[0].at<float>(BGR_Image[0].at<uchar>(row, col));
      float G = vCumHist[1].at<float>(BGR_Image[1].at<uchar>(row, col));
      float R = vCumHist[2].at<float>(BGR_Image[2].at<uchar>(row, col));

      BGR_Image[0].at<uchar>(row, col) =
          cv::saturate_cast<uchar>(255 * std::pow(alpha, B));
      BGR_Image[1].at<uchar>(row, col) =
          cv::saturate_cast<uchar>(255 * std::pow(alpha, G));
      BGR_Image[2].at<uchar>(row, col) =
          cv::saturate_cast<uchar>(255 * std::pow(alpha, R));
    }
  }
  merge(BGR_Image, img);

  for (auto& i : vHist) i.setTo(cv::Scalar::all(0));
  for (auto& i : vCumHist) i.setTo(cv::Scalar::all(0));

  histogram(img, vHist, vCumHist);
}
void lab1::profile(const cv::Mat& img, cv::Mat& pro_x, cv::Mat& pro_y) {
  cv::Mat gray_img;
  cv::cvtColor(img, gray_img, CV_BGR2GRAY);
  pro_x = gray_img.row(gray_img.rows / 2);
  pro_y = gray_img.col(gray_img.cols / 2);
}
void lab1::projection(const cv::Mat& img, cv::Mat& proj_x, cv::Mat& proj_y) {
  cv::Mat gray_img;
  cv::cvtColor(img, gray_img, CV_BGR2GRAY);
  // Sum by Rows - ProjY
  cv::reduce(gray_img, proj_y, 1, cv::REDUCE_SUM, CV_32F);
  // Sum by Cols - ProjX
  cv::reduce(gray_img, proj_x, 0, cv::REDUCE_SUM, CV_32F);
  proj_x /= 255;
  proj_y /= 255;
}
}  // namespace DIP