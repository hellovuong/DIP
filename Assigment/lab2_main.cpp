//
// Created by vuong on 01/12/2021.
//
#include <iostream>
#include "lab2.h"

int main(int argc, char** argv) {
  if (argc <= 4) {
    std::cerr << std::endl
              << "Usage ./lab2_main /path/to/image/ /path/to/image1/ "
                 "/path/to/image_left/ /path/to/image_right/"
              << std::endl;
    return EXIT_FAILURE;
  }

  const cv::Mat orig_img = cv::imread(std::string(argv[1]));
  const cv::Mat dist_img = cv::imread(std::string(argv[2]));
  const cv::Mat left_img = cv::imread(std::string(argv[3]));
  const cv::Mat right_img = cv::imread(std::string(argv[4]));
  cv::Mat undist_img;

  if (orig_img.empty() || dist_img.empty() || left_img.empty() ||
      right_img.empty()) {
    std::cerr << std::endl << "Image can not found" << std::endl;
    return EXIT_FAILURE;
  }

  cv::Mat new_img, show_img;
  DIP::lab2::shift_image(orig_img, new_img);
  cv::hconcat(orig_img, new_img, show_img);
  cv::imshow("shifted image", show_img);
  cv::waitKey(0);
  cv::imwrite("shifted.png", show_img);
  show_img.setTo(0);
  new_img.setTo(0);

  DIP::lab2::flip_image(orig_img, new_img);
  cv::hconcat(orig_img, new_img, show_img);
  cv::imshow("flipped image", show_img);
  cv::waitKey(0);
  cv::imwrite("flipped.png", show_img);
  show_img.setTo(0);
  new_img.setTo(0);

  DIP::lab2::rotate_image(20.0, orig_img, new_img);
  cv::hconcat(orig_img, new_img, show_img);
  cv::imshow("rotated image", show_img);
  cv::waitKey(0);
  cv::imwrite("rotated.png", show_img);
  show_img.setTo(0);
  new_img.setTo(0);

  DIP::lab2::uniform_scale(3, orig_img, new_img);
  //  cv::hconcat(orig_img, new_img, show_img);
  cv::imshow("scaled image", new_img);
  cv::waitKey(0);
  cv::imwrite("scaled.png", new_img);
  show_img.setTo(0);

  new_img = cv::Mat::zeros(orig_img.size(), orig_img.type());
  DIP::lab2::affine2d_image(orig_img, new_img);
  cv::hconcat(orig_img, new_img, show_img);
  cv::imshow("affine2d image", show_img);
  cv::waitKey(0);
  cv::imwrite("affined.png", show_img);
  show_img.setTo(0);

  new_img = cv::Mat::zeros(orig_img.size(), orig_img.type());
  DIP::lab2::bevel_image(0.5, orig_img, new_img);
  cv::hconcat(orig_img, new_img, show_img);
  cv::imshow("bevel image", show_img);
  cv::waitKey(0);
  cv::imwrite("beveled.png", show_img);
  show_img.setTo(0);

  // Piecewise transformation
  new_img = cv::Mat::zeros(orig_img.size(), orig_img.type());
  DIP::lab2::flip_pw(orig_img, new_img);
  cv::hconcat(orig_img, new_img, show_img);
  cv::imshow("piecewise flip image", show_img);
  cv::waitKey(0);
  cv::imwrite("pw_flip.png", show_img);
  show_img.setTo(0);

  // Projective mapping
  new_img = cv::Mat::zeros(orig_img.size(), orig_img.type());
  DIP::lab2::projective_mapping(orig_img, new_img);
  //  cv::hconcat(orig_img, new_img, show_img);
  cv::imshow("projective-mapping", new_img);
  cv::imwrite("projective.png", new_img);
  cv::waitKey(0);
  show_img.setTo(0);

  // Polynomial Mapping
  new_img = cv::Mat::zeros(orig_img.size(), orig_img.type());
  DIP::Vector6d A, B;
  A << 0, 1, 0, 0.00001, 0.002, 0.002;
  B << 0, 0, 1, 0, 0, 0;
  DIP::lab2::polynomial_mapping(A, B, orig_img, new_img);
  //  cv::hconcat(orig_img, new_img, show_img);
  cv::imshow("pol image", new_img);
  cv::waitKey(0);
  cv::imwrite("pol.png", new_img);
  show_img.setTo(0);

  // Sinusoidal distortion
  new_img = cv::Mat::zeros(orig_img.size(), orig_img.type());
  DIP::lab2::sinusoidal_distortion(orig_img, new_img);
  cv::hconcat(orig_img, new_img, show_img);
  cv::imshow("Sinusoidal distortion", show_img);
  cv::waitKey(0);
  cv::imwrite("sin_dist.png", show_img);
  show_img.setTo(0);

  // fisheye undistortion
  DIP::lab2::undistort_fisheye(dist_img, undist_img);
  cv::hconcat(dist_img, undist_img, show_img);
  cv::imshow("fisheye undistortion", show_img);
  cv::waitKey(0);
  cv::imwrite("fisheye_und.png", show_img);

  // Automatic panorama stitching
  std::vector<cv::Mat> imgs;
  imgs.push_back(left_img);
  imgs.push_back(right_img);
  cv::Mat pano;
  DIP::lab2::pano_stitcher(imgs, pano);
  cv::imshow("Automatic panorama stitching", pano);
  cv::waitKey(0);
  cv::imwrite("pano.png", pano);

  return EXIT_SUCCESS;
}