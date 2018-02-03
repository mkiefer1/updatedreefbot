#ifndef __GEOMETRIC_TRANSFORMS_INL_H__
#define __GEOMETRIC_TRANSFORMS_INL_H__

#include "GeometricTransforms.h"

#include <ros/ros.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

using cv::Mat_;
using cv::Mat;

namespace geo {

// AffineWarp
//
// Warps an image using a 2x3 affine transformation matrix and
// bilinear interpolation.
//
// Areas outside of the frame are left as the same value as the original image
//
// Inputs:
// image - The image to warp
// matrix - 2x3 transformation matrix
//
// Returns: A warped version of the matrix
template<typename T>
Mat_<T> AffineWarp(const Mat_<T>& image, const Mat& M,
                   BorderFillType borderFillType,
                   const T& fillConstant,
                   const Mat_<T>& backgroundImage) {

  Mat_<T> retval;
  if (borderFillType == geo::BORDER_TRANSPARENT) {
    if (image.rows != backgroundImage.rows ||
        image.cols != backgroundImage.cols) {
      ROS_FATAL("Background must be the same size as the image");
      return retval;
    }
    retval = backgroundImage.clone();
  } else if (borderFillType == geo::BORDER_CONSTANT) {
    retval = Mat_<T>(image.rows, image.cols, fillConstant);
  } else {
    ROS_FATAL("Invalid fill type");
    return retval;
  }

  // We use the inverse matrix because we will fill the output matrix
  // by finding the correct spot in the input matrix. This requries an
  // inversion.
  if (M.rows != 2 && M.cols != 3) {
    ROS_FATAL("Transformation matrix must be 2x3");
    return retval;
  }
  Mat_<double> invM;
  cv::invertAffineTransform(M, invM);
  const double* ptrM = invM.ptr<double>(0);

  if (!retval.isContinuous()) {
    ROS_FATAL("Image is not continuous. It must be for speed reasons.");
    return retval;
  }

  T* curPixel = reinterpret_cast<T*>(retval.data);
  for (int y=0; y < image.rows; y++) {
    for (int x=0; x < image.cols; x++, curPixel++) {

      // First calculate the x and y coordinates to sample from for this pixel
      const double newX = ptrM[0]*x + ptrM[1]*y + ptrM[2];
      const double newY = ptrM[3]*x + ptrM[4]*y + ptrM[5];

      // Make sure the coorindate is valid.
      if (newX < 0 || newY < 0 || 
          newX > (image.cols-1) || newY > (image.rows-1)) {
        continue;
      }

      BilinearInterpPoint(image, newX, newY, curPixel);
    }
  }

  return retval;
}

template <typename T>
inline
void BilinearInterpPoint(const Mat_<T>& image,
                         const double x,
                         const double y,
                         T* output) {
  const int x1 = static_cast<int>(x);
  const int x2 = static_cast<int>(ceil(x));
  const int y1 = static_cast<int>(y);
  const int y2 = static_cast<int>(ceil(y));

  const double x2_x = x2-x;
  const double x_x1 = x-x1;
  const double y2_y = y2-y;
  const double y_y1 = y-y1;

  if (x1 != x2 && y1 != y2) {
    *output = y2_y * (x2_x * image(y1, x1) + x_x1 * image(y1, x2)) +
      y_y1 * (x2_x * image(y2, x1) + x_x1 * image(y2, x2));
  } else if (x1 == x2 && y1 == y2) {
    // At a corner
    *output = image(y1, x1);
  } else if (x1 == x2) {
    // On the edge where x is consistent
    *output = image(y2,x1)*y_y1 - image(y1,x1)*(y_y1-1);
  } else if (y1 == y2) {
    // On the edge where y is consistent
    *output = image(y1,x2)*x_x1 - image(y1,x1)*(x_x1-1);
  } else {
    ROS_FATAL("Shouldn't get here");
  }
}

} // namespace

#endif // __GEOMETRIC_TRANSFORMS_INL_H__
