//
// Created by vuong on 02/12/2021.
//

#include "lab3.h"
#include <iostream>
#include <random>
namespace DIP {
void DIP::lab3::impluse_noise(const cv::Mat& im, cv::Mat& noise_im) {
  cv::Mat saltpepper_noise = cv::Mat::zeros(im.rows, im.cols, CV_8U);
  cv::randu(saltpepper_noise, 0, 255);

  cv::Mat black = saltpepper_noise < 30;
  cv::Mat white = saltpepper_noise > 225;

  noise_im = im.clone();
  noise_im.setTo(255, white);
  noise_im.setTo(0, black);
}
void lab3::speckle_noise(const cv::Mat& im, cv::Mat& noise_im) {
  noise_im = im.clone();
  int size[3] = {im.rows, im.cols, im.dims};
  cv::Mat mult_noise(im.dims, size, im.type());
  cv::theRNG().fill(mult_noise, cv::RNG::NORMAL, 0, 1);

  cv::multiply(noise_im, mult_noise, noise_im);
}
void lab3::gaussian_noise(const double mean, const double stddev,
                          const cv::Mat& im, cv::Mat& noise_im) {
  int size[3] = {im.rows, im.cols, im.dims};
  cv::Mat gaussian_noise(im.dims, size, im.type());
  noise_im = im.clone();
  cv::randn(gaussian_noise, mean, stddev);
  cv::addWeighted(noise_im, 1.0, gaussian_noise, 1.0, 0.0, noise_im);
}
void lab3::poisson_noise(const cv::Mat& im, cv::Mat& noise_im) {}
void lab3::gaussian_filter(cv::Mat& noise_im, cv::Mat& denoise_im) {
  cv::GaussianBlur(noise_im, denoise_im, cv::Size(3, 3), 0);
}
void lab3::couterHarmonic_filter(cv::Mat& noise_im, cv::Mat& denoise_im,
                                 int kSize, double Q) {
  cv::Mat _src;
  //  cvtColor(noise_im, denoise_im, CV_BGR2GRAY);
  copyMakeBorder(noise_im, _src, (kSize / 2), (kSize / 2), (kSize / 2),
                 (kSize / 2), cv::BORDER_REPLICATE);

  float val;

  for (int i = (kSize / 2); i < _src.rows - (kSize / 2); i++) {
    for (int j = (kSize / 2); j < _src.cols - (kSize / 2); j++) {
      float a = 0;
      float b = 0;

      for (int k = i - (kSize / 2); k <= i + (kSize / 2); k++) {
        for (int w = j - (kSize / 2); w <= j + (kSize / 2); w++) {
          val = (float)_src.at<uchar>(k, w);
          a += (float)std::pow(val, (1.f + Q));
          b += (float)std::pow(val, Q);
        }
      }

      denoise_im.at<uchar>(i - (kSize / 2), j - (kSize / 2)) =
          cv::saturate_cast<uchar>(a / b);
    }
  }
}
void lab3::median_filter(cv::Mat& noise_im, cv::Mat& denoise_im, int kSize) {
  medianBlur(noise_im, denoise_im, kSize);
}
void lab3::weightedMedian_filter(cv::Mat& noise_im, cv::Mat& denoise_im,
                                 int kSize) {
  cv::ximgproc::weightedMedianFilter(noise_im, noise_im, denoise_im, kSize);
}
void lab3::rank_filter(const std::string& method, int kSize, cv::Mat& noise_im,
                       cv::Mat& denoise_im) {
  int iMethod;
  if (method == "min") iMethod = 0;
  if (method == "max") iMethod = 1;

  cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT,
                                             cv::Size(kSize, kSize));
  switch (iMethod) {
    case 0: {
      cv::erode(noise_im, denoise_im, kernel);
      break;
    }
    case 1: {
      cv::dilate(noise_im, denoise_im, kernel);
      break;
    }
    default:
      std::cout << "Default " << std::endl;
  }
}
uchar lab3::adaptive_iterate(const cv::Mat& im, int row, int col,
                             int kernelSize, int maxSize) {
  std::vector<uchar> pixels;
  for (int a = -kernelSize / 2; a <= kernelSize / 2; a++) {
    for (int b = -kernelSize / 2; b <= kernelSize / 2; b++) {
      pixels.push_back(im.at<uchar>(row + a, col + b));
    }
  }
  sort(pixels.begin(), pixels.end());
  auto min = pixels[0];
  auto max = pixels[kernelSize * kernelSize - 1];
  auto med = pixels[kernelSize * kernelSize / 2];
  auto zxy = im.at<uchar>(row, col);
  if (med > min && med < max) {
    if (zxy > min && zxy < max) {
      return zxy;
    } else {
      return med;
    }
  } else {
    kernelSize += 2;
    if (kernelSize <= maxSize)
      return adaptive_iterate(im, row, col, kernelSize, maxSize);
    else
      return med;
  }
}
void lab3::adaptive_filter(int minSize, int maxSize, cv::Mat& noise_im,
                           cv::Mat& denoise_im) {
  copyMakeBorder(noise_im, denoise_im, maxSize / 2, maxSize / 2, maxSize / 2,
                 maxSize / 2, cv::BORDER_REFLECT);
  int rows = denoise_im.rows;
  int cols = denoise_im.cols;
  for (int j = maxSize / 2; j < rows - maxSize / 2; j++) {
    for (int i = maxSize / 2; i < cols * denoise_im.channels() - maxSize / 2;
         i++) {
      denoise_im.at<uchar>(j, i) =
          adaptive_iterate(denoise_im, j, i, minSize, maxSize);
    }
  }
}

// G_{x} = |1  -1|    G_{y} = | 1   0|
//         |0   0|            |-1   0|
//
void lab3::roberts_detector(const cv::Mat& im, cv::Mat& dst) {
  // reduce noise
  cv::Mat denoise_im;
  cv::GaussianBlur(im, denoise_im, cv::Size(3, 3), 0);
  dst = denoise_im.clone();

  // Roberts operator
  int nRows = dst.rows;
  int nCols = dst.cols;
  for (int i = 0; i < nRows - 1; i++) {
    for (int j = 0; j < nCols - 1; j++) {
      int t1 = (im.at<uchar>(i, j) - im.at<uchar>(i + 1, j + 1)) *
               (im.at<uchar>(i, j) - im.at<uchar>(i + 1, j + 1));
      int t2 = (im.at<uchar>(i + 1, j) - im.at<uchar>(i, j + 1)) *
               (im.at<uchar>(i + 1, j) - im.at<uchar>(i, j + 1));
      dst.at<uchar>(i, j) = (uchar)sqrt(t1 + t2);
    }
  }
}
//         |-1  0  1|            | -1  -1  -1|
// G_{x} = |-1  0  1|    G_{y} = | 0   0    0|
//         |-1  0  1|            | 1   1    1|
//
void lab3::prewitt_detector(const cv::Mat& im, cv::Mat& dst) {
  // reduce noise
  cv::Mat denoise_im;
  cv::GaussianBlur(im, denoise_im, cv::Size(3, 3), 0);

  // Prewitt operator
  cv::Mat Gx = (cv::Mat_<int>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
  cv::Mat Gy = (cv::Mat_<int>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
  cv::Mat dst_x, dst_y;
  cv::filter2D(denoise_im, dst_x, -1, Gx);
  cv::filter2D(denoise_im, dst_y, -1, Gy);

  // Merge image
  cv::addWeighted(dst_x, 0.5, dst_y, 0.5, 0.0, dst);
}
void lab3::sobel_detector(int ksize, const cv::Mat& im, cv::Mat& dst) {
  // reduce noise
  cv::Mat denoise_im;
  cv::GaussianBlur(im, denoise_im, cv::Size(3, 3), 0);

  // Sobel operator
  cv::Mat grad_x, grad_y;
  cv::Mat abs_grad_x, abs_grad_y;
  cv::Sobel(denoise_im, grad_x, CV_16S, 1, 0, ksize, 1, 0, cv::BORDER_DEFAULT);
  cv::Sobel(denoise_im, grad_y, CV_16S, 0, 1, ksize, 1, 0, cv::BORDER_DEFAULT);

  // converting back to CV_8U
  cv::convertScaleAbs(grad_x, abs_grad_x);
  cv::convertScaleAbs(grad_y, abs_grad_y);

  // Merge image
  cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
}
void lab3::laplacian_detector(int kernel_size, const cv::Mat& im,
                              cv::Mat& dst) {
  // reduce noise
  cv::Mat denoise_im;
  cv::GaussianBlur(im, denoise_im, cv::Size(3, 3), 0);

  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  // Laplacian operator
  cv::Laplacian(denoise_im, dst, ddepth, kernel_size, scale, delta,
            cv::BORDER_DEFAULT);

  // converting back to CV_8U
  cv::convertScaleAbs(dst, dst);
}
void lab3::canny_detector(int low_threshold, int kernel_size, const cv::Mat& im, cv::Mat& dst) {

  const int ratio = 3;

  // reduce noise
  cv::Mat denoise_im;
  cv::GaussianBlur(im, denoise_im, cv::Size(3, 3), 0);

  cv::Canny( denoise_im, dst, low_threshold, low_threshold*ratio, kernel_size );
}

}  // namespace DIP
