#ifndef __GEOMETRIC_TRANSFORMS_H__
#define __GEOMETRIC_TRANSFORMS_H__

#include <opencv2/core/core.hpp>

namespace geo {

// Specifies what to do for locations in the image outside of the
// frame when warping.
typedef enum {
  BORDER_TRANSPARENT=0, // Keeps the same value as in the original image
  BORDER_CONSTANT=1 // Fill with a constant value
} BorderFillType;

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
// borderFillType - Defines how to fill the areas outside of the frame
// fillConstant - For BORDER_CONSTANT, the value to fill with
// backgroundImage - For BORDER_TRANSPARENT, the image to fill with. Must be the same size as image
//
// Returns: A warped version of the matrix
template<typename T>
cv::Mat_<T> AffineWarp(const cv::Mat_<T>& image, const cv::Mat& M,
                       BorderFillType borderFillType,
                       const T& fillConstant=T(),
                       const cv::Mat_<T>& backgroundImage=cv::Mat_<T>());

// Performs bilinear interpolation in an image of a single point.
//
// There is no checking that x and y are in the proper range
template<typename T>
inline
void BilinearInterpPoint(const cv::Mat_<T>& image,
                         const double x,
                         const double y,
                         T* output);

} // namespace


#endif // __GEOMETRIC_TRANSFORMS_H__
