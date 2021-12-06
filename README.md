# Digtal Image Processing
Repo for Digital Image Processing ITMO 2021

Lab 1: Image Histograms, Profiles, Projections
- Histogram: Create Histogram, Histogram Arithmetic,Dynamic Range Stretching, Uniform, Rayleigh, Exponential, 2/3 degree, Hyperbolic Transformation.
- Histogram Equalization
- Image Profile
- Image Projection

Lab 2: Images Geometric Transformations
- Linear Mapping: Shift, Flip, Rotate, Affine, etc.
- Non-linear Mapping: Projection, Polynomial.
- Distortion Correction
- Panorama Creation

Lab 3: Filtering and Edges Detection
- Noise Creation: Impulse, Additive, Gussian, etc.
- Filter: Low-pass, Nonlinear Filter.
- Edge Detector: Roberts, Prewitt, Sobel, Laplace, Canny.

Detail implementation: res/*/(308560)Long Vuong.pdf
# Prerequisites
- OpenCV 3
- Eigen3
- Python 2.7
# Installation
```sh
git clone --recursive https://github.com/hellovuong/DIP.git
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j2
```

# Run

Histogram Operation
```sh
./Assigment/lab1_main /path/to/image
```
Image transformation and undistortion
```sh
./Assigment/lab2_main /path/to/image /path/to/image/to/undistort /path/to/left/image path/to/right/image
```
Create noise and denosie image
```sh
./Assigment/lab3_main /path/to/image
```