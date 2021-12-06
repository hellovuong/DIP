//
// Created by vuong on 30/11/2021.
//
#include <iostream>
#include "lab1.h"

int main(int argc, char** argv) {
  if (argc <= 1) {
    std::cerr << std::endl << "Usage ./lab1_main /path/to/image/" << std::endl;
    return EXIT_FAILURE;
  }
  cv::Mat orig_img = cv::imread(std::string(argv[1]), CV_LOAD_IMAGE_COLOR);
  if (orig_img.empty()) {
    std::cerr << std::endl << "Image can not found" << std::endl;
    return EXIT_FAILURE;
  }
  const int hist_w = 512, hist_h = orig_img.rows;
  auto* lab1 = new DIP::lab1();
  std::vector<cv::Mat> cum_hist;
  cum_hist.resize(3);
  cv::Mat new_img = orig_img.clone();
  // Create histogram
  cv::Mat show_im;
  std::vector<cv::Mat> vHist;
  vHist.resize(3);
  lab1->histogram(orig_img, vHist, cum_hist);
  cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));
  histImage.setTo(cv::Scalar(255, 255, 255));
  lab1->plot_hist(histImage, vHist);
  cv::hconcat(orig_img, histImage, show_im);
  cv::imshow("Histogram", show_im);
  cv::imwrite("./res/lab1/orig_hist.png", show_im);
  cv::cvtColor(new_img, new_img, CV_BGR2GRAY);
  cv::resize(new_img, new_img, cv::Size(640, 480));

  cv::imwrite("./res/lab1/gray.png", new_img);
  cv::waitKey();

  // Arithmetics Operation
  orig_img.convertTo(new_img, -1, 1, 50);
  lab1->histogram(new_img, vHist, cum_hist);
  cv::hconcat(new_img, histImage, show_im);
  cv::imshow("Histogram Arth", show_im);
  cv::imwrite("./res/lab1/arith.png", show_im);
  cv::waitKey();

  // Dynamic Range Stretching
  new_img = orig_img.clone();
  for (auto& i : vHist) i.setTo(cv::Scalar::all(0));

  cv::Mat new_histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));
  new_histImage.setTo(cv::Scalar(255, 255, 255));

  lab1->drs(new_img, vHist, 0.7);
  lab1->plot_hist(new_histImage, vHist);

  cv::Mat show_im2;
  //  cv::hconcat(histImage, new_histImage, show_im);
  //  cv::hconcat(orig_img, new_img, show_im2);

  //  cv::imshow("Histogram", show_im);
  //  cv::waitKey();
  //  cv::imshow("Image", show_im2);
  //  cv::waitKey();
  cv::hconcat(new_img, new_histImage, show_im);
  cv::imshow("Histogram DRS", show_im);
  cv::imwrite("./res/lab1/drs.png", show_im);
  cv::waitKey();

  // Uniform transformation
  new_img = orig_img.clone();
  for (auto& i : vHist) i.setTo(cv::Scalar::all(0));
  new_histImage.setTo(cv::Scalar(255, 255, 255));

  lab1->uniform_trans(new_img, vHist);
  lab1->plot_hist(new_histImage, vHist);

  //  cv::hconcat(histImage, new_histImage, show_im);
  //  cv::hconcat(orig_img, new_img, show_im2);

  //  cv::imshow("Histogram", show_im);
  //  cv::waitKey();
  //  cv::imshow("Image", show_im2);
  //  cv::waitKey();
  cv::hconcat(new_img, new_histImage, show_im);
  cv::imshow("Histogram", show_im);
  cv::imwrite("./res/lab1/uniform.png", show_im);
  cv::waitKey();

  // Exponential Transform
  new_img = orig_img.clone();
  for (auto& i : vHist) i.setTo(cv::Scalar::all(0));
  new_histImage.setTo(cv::Scalar(255, 255, 255));

  lab1->exp_trans(new_img, vHist, 0.1);
  lab1->plot_hist(new_histImage, vHist);

  //  cv::hconcat(histImage, new_histImage, show_im);
  //  cv::hconcat(orig_img, new_img, show_im2);

  //  cv::imshow("Histogram", show_im);
  //  cv::waitKey();
  //  cv::imshow("Image", show_im2);
  //  cv::waitKey();
  cv::hconcat(new_img, new_histImage, show_im);
  cv::imshow("Histogram", show_im);
  cv::imwrite("./res/lab1/exp.png", show_im);
  cv::waitKey();

  // Ray trans
  new_img = orig_img.clone();
  for (auto& i : vHist) i.setTo(cv::Scalar::all(0));
  new_histImage.setTo(cv::Scalar(255, 255, 255));

  lab1->ray_trans(new_img, vHist, 0.1);
  lab1->plot_hist(new_histImage, vHist);

  //  cv::hconcat(histImage, new_histImage, show_im);
  //  cv::hconcat(orig_img, new_img, show_im2);

  //  cv::imshow("Histogram", show_im);
  //  cv::waitKey();
  //  cv::imshow("Image", show_im2);
  //  cv::waitKey();
  cv::hconcat(new_img, new_histImage, show_im);
  cv::imshow("Histogram", show_im);
  cv::imwrite("./res/lab1/ray.png", show_im);
  cv::waitKey();

  // Trans 2/3
  new_img = orig_img.clone();
  for (auto& i : vHist) i.setTo(cv::Scalar::all(0));
  new_histImage.setTo(cv::Scalar(255, 255, 255));

  lab1->trans23(new_img, vHist);
  lab1->plot_hist(new_histImage, vHist);

  //  cv::hconcat(histImage, new_histImage, show_im);
  //  cv::hconcat(orig_img, new_img, show_im2);

  //  cv::imshow("Histogram", show_im);
  //  cv::waitKey();
  //  cv::imshow("Image", show_im2);
  //  cv::waitKey();
  cv::hconcat(new_img, new_histImage, show_im);
  cv::imshow("Histogram", show_im);
  cv::imwrite("./res/lab1/trans2_3.png", show_im);
  cv::waitKey();

  // Hyperbolic Trans
  new_img = orig_img.clone();
  for (auto& i : vHist) i.setTo(cv::Scalar::all(0));
  new_histImage.setTo(cv::Scalar(255, 255, 255));

  lab1->hyper_trans(new_img, vHist, 0.05);
  lab1->plot_hist(new_histImage, vHist);

  //  cv::hconcat(histImage, new_histImage, show_im);
  //  cv::hconcat(orig_img, new_img, show_im2);

  //  cv::imshow("Histogram", show_im);
  //  cv::waitKey();
  //  cv::imshow("Image", show_im2);
  //  cv::waitKey();
  cv::hconcat(new_img, new_histImage, show_im);
  cv::imshow("Histogram", show_im);
  cv::imwrite("./res/lab1/hyper.png", show_im);
  cv::waitKey();

  // CLAHE
  cv::Mat orig_gray_img, new_gray_img;
  cv::cvtColor(orig_img, orig_gray_img, CV_BGR2GRAY);
  new_gray_img = orig_gray_img.clone();
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
  clahe->setClipLimit(4);
  clahe->apply(orig_gray_img, new_gray_img);
  cv::hconcat(orig_gray_img, new_gray_img, show_im2);
  cv::imshow("Image", show_im2);
  cv::waitKey();
  cv::imwrite("./res/lab1/clahe.png", show_im2);

  // Image Profile
  cv::Mat profile_ox, profile_oy;
  DIP::lab1::profile(orig_img, profile_ox, profile_oy);
  std::vector<int> cols, pix_x, rows, pix_y;
  for (int i = 0; i < profile_ox.cols; ++i) {
    cols.push_back(i);
    pix_x.push_back((int)profile_ox.at<uchar>(0, i));
  }
  for (int i = 0; i < profile_oy.rows; ++i) {
    rows.push_back(i);
    pix_y.push_back((int)profile_oy.at<uchar>(0, i));
  }

  matplotlibcpp::plot(cols, pix_x);
  matplotlibcpp::xlabel("x (pixel)");
  matplotlibcpp::ylabel("Intensity");
  matplotlibcpp::xlim(0, profile_ox.cols);
  matplotlibcpp::figure_size(512, 512);
  matplotlibcpp::show(true);
  matplotlibcpp::plot(rows, pix_y);
  matplotlibcpp::xlabel("y (pixel)");
  matplotlibcpp::ylabel("Intensity");
  matplotlibcpp::xlim(0, profile_oy.rows);
  matplotlibcpp::figure_size(512, 512);
  matplotlibcpp::show(true);
  //  matplotlibcpp::save("./res/lab1/profile_x.png",255);

  // Image Projection
  cv::Mat proj_oy;
  cv::Mat proj_ox;

  DIP::lab1::projection(orig_img, proj_ox, proj_oy);
  std::vector<double> proj_x, proj_y, vCol, vRow;
  for (int i = 0; i < proj_ox.cols; ++i) {
    vCol.push_back(i);
    proj_x.push_back(proj_ox.at<float>(0, i));
  }
  for (int i = 0; i < proj_oy.rows; ++i) {
    vRow.push_back(i);
    proj_y.push_back(proj_oy.at<float>(i, 0));
  }
  matplotlibcpp::figure_size((int)proj_x.size(), (int)proj_x.size());
  matplotlibcpp::plot(vCol, proj_x);
  matplotlibcpp::title("Projection on X axis");
  matplotlibcpp::xlabel("x (pixel)");
  matplotlibcpp::ylabel("Intensity");
  matplotlibcpp::xlim(0, (int)proj_x.size());
  matplotlibcpp::show(true);
  //  matplotlibcpp::save("./res/lab1/proj_x.png",255);

  matplotlibcpp::xlim(0, (int)proj_y.size());
  matplotlibcpp::plot(vRow, proj_y);
  matplotlibcpp::title("Projection on Y axis");
  matplotlibcpp::xlabel("y (pixel)");
  matplotlibcpp::ylabel("Intensity");
  matplotlibcpp::figure_size((int)proj_y.size(), (int)proj_y.size());
  matplotlibcpp::show(true);
  //  matplotlibcpp::save("./res/lab1/profile_y.png",255);

  return EXIT_SUCCESS;
}