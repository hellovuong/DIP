//
// Created by vuong on 02/12/2021.
//
#include <iostream>
#include "lab3.h"

int main(int argc, char** argv) {
  if (argc <= 1) {
    std::cerr << std::endl << "Usage ./lab3_main /path/to/image/" << std::endl;
    return EXIT_FAILURE;
  }
  const cv::Mat orig_im =
      cv::imread(std::string(argv[1]), cv::IMREAD_GRAYSCALE);
  cv::Mat im, new_im, show_im;

  //  orig_im.convertTo(im, CV_8UC3);
  im = orig_im.clone();
  cv::imwrite("gray.png",im);
  if (im.empty()) {
    std::cerr << std::endl << "Image can not found" << std::endl;
    return EXIT_FAILURE;
  }

  // Impluse noise
  DIP::lab3::impluse_noise(im, new_im);
  cv::hconcat(im, new_im, show_im);
  cv::imshow("salt & peper noise", show_im);
  cv::imwrite("salt_peper.png",show_im);
  cv::waitKey(0);

  // Speckle noise
  DIP::lab3::speckle_noise(im, new_im);
  cv::hconcat(im, new_im, show_im);
  cv::imshow("speckle noise", show_im);
  cv::imwrite("speckle.png",show_im);
  cv::waitKey(0);

  // Gaussian noise
  DIP::lab3::gaussian_noise(20, 30.0, im, new_im);
  cv::hconcat(im, new_im, show_im);
  cv::imshow("gaussian noise", show_im);
  cv::imwrite("gaus_noise.png",show_im);
  cv::waitKey(0);
  cv::Mat denoise_im;
  DIP::lab3::gaussian_filter(new_im, denoise_im);
  cv::hconcat(new_im, denoise_im, show_im);
  cv::imshow("gaussian denoise", show_im);
  cv::imwrite("gaus_denoise.png",show_im);
  cv::waitKey(0);

  cv::Mat new_show, gray_new_im;
  DIP::lab3::couterHarmonic_filter(new_im, denoise_im, 3, -1.5f);
  //  cv::cvtColor(new_im, gray_new_im, CV_BGR2GRAY);
  cv::hconcat(new_im, denoise_im, new_show);
  cv::imshow("CH denoise", new_show);
  cv::imwrite("ch_denoise.png",show_im);
  cv::waitKey(0);

  DIP::lab3::median_filter(new_im, denoise_im, 3);
  cv::hconcat(new_im, denoise_im, show_im);
  cv::imshow("median denoise", show_im);
  cv::imwrite("median_denoise.png",show_im);
  cv::waitKey(0);

  DIP::lab3::weightedMedian_filter(new_im, denoise_im, 3);
  cv::hconcat(new_im, denoise_im, show_im);
  cv::imshow("weighted median denoise", show_im);
  cv::imwrite("weighted_ median_denoise.png",show_im);
  cv::waitKey(0);

  DIP::lab3::rank_filter("min", 3, new_im, denoise_im);
  cv::hconcat(new_im, denoise_im, show_im);
  cv::imshow("min-filter denoise", show_im);
  cv::imwrite("min_denoise.png",show_im);
  cv::waitKey(0);

  DIP::lab3::rank_filter("max", 3, new_im, denoise_im);
  cv::hconcat(new_im, denoise_im, show_im);
  cv::imshow("max-filter denoise", show_im);
  cv::imwrite("max_denoise.png",show_im);
  cv::waitKey(0);

  DIP::lab3::adaptive_filter(3, 7, new_im, denoise_im);
  cv::resize(denoise_im, denoise_im, new_im.size());
  cv::hconcat(new_im, denoise_im, show_im);
  cv::imshow("adaptive filter denoise", show_im);
  cv::imwrite("adap_denoise.png",show_im);
  cv::waitKey(0);

  DIP::lab3::roberts_detector(im, new_im);
  cv::hconcat(im, new_im, show_im);
  cv::imshow("Roberts Edge Detector", show_im);
  cv::imwrite("rot_det.png",show_im);
  cv::waitKey(0);

  DIP::lab3::prewitt_detector(im, new_im);
  cv::hconcat(im, new_im, show_im);
  cv::imshow("Prewitt Edge Detector", show_im);
  cv::imwrite("prewitt_det.png",show_im);
  cv::waitKey(0);

  DIP::lab3::sobel_detector(3, im, new_im);
  cv::hconcat(im, new_im, show_im);
  cv::imshow("Sobel Edge Detector", show_im);
  cv::imwrite("sobel.png",show_im);
  cv::waitKey(0);

  DIP::lab3::laplacian_detector(3, im, new_im);
  cv::hconcat(im, new_im, show_im);
  cv::imshow("Laplacian Edge Detector", show_im);
  cv::imwrite("lap.png",show_im);
  cv::waitKey(0);

  DIP::lab3::canny_detector(50, 3, im, new_im);
  cv::hconcat(im, new_im, show_im);
  cv::imshow("Canny Edge Detector", show_im);
  cv::imwrite("canny.png",show_im);
  cv::waitKey(0);
  return EXIT_SUCCESS;
}
