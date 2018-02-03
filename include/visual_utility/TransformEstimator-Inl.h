#include "TransformEstimator.h"

#include "GeometricTransforms-Inl.h"

using namespace cv;

namespace visual_utility {

template <typename T>
Mat_<T> AffineTransformEstimator::ApplyTransformImpl(
  const Mat_<T>& image,
  const Mat& transform,
  geo::BorderFillType borderFillType,
  const T& fillConstant,
  const cv::Mat_<T>& backgroundImage) const {
  return geo::AffineWarp(image, transform, borderFillType, fillConstant,
                         backgroundImage);
}

} // namespace
