// Estimators for figuring out the transform between two images.
//
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
// Date: July 2011
#ifndef __TRANSFORM_ESTIMATOR_H__
#define __TRANSFORM_ESTIMATOR_H__

#include <opencv2/core/core.hpp>
#include "GeometricTransforms.h"

namespace visual_utility {

// Abtract transform estimator class
class TransformEstimator {
public:

  TransformEstimator() {}

  virtual ~TransformEstimator();

  // Estimate the transform between image1 and image 2
  //
  // The return value is .empty() if it fails
  virtual cv::Mat EstimateTransform(const cv::Mat_<double>& image1,
                                    const cv::Mat_<double>& image2) const=0;

  // Apply a transform to an image
  //
  // Inputs:
  // image - image to transform
  // transform - transform to apply
  //
  // Returns the transformed image
  virtual
  cv::Mat_<double> ApplyTransform(
    const cv::Mat_<double>& image,
    const cv::Mat transform,
    geo::BorderFillType borderFillType,
    const double& fillConstant = 0.0,
    const cv::Mat_<double>& backgroundImage=cv::Mat_<double>()) const =0;

  virtual
  cv::Mat_<cv::Vec3d> ApplyTransform(
    const cv::Mat_<cv::Vec3d>& image,
    const cv::Mat transform,
    geo::BorderFillType borderFillType,
    const cv::Vec3d& fillConstant=cv::Vec3d(0,0,0),
    const cv::Mat_<cv::Vec3d>& backgroundImage=cv::Mat_<cv::Vec3d>()) const =0;

};

class AffineTransformEstimator : public TransformEstimator {
public:
  AffineTransformEstimator(int maxIterations, double minPrecision,
                           double imageScaling) 
    : TransformEstimator(), maxIterations_(maxIterations),
      minPrecision_(minPrecision), imageScaling_(imageScaling) {}

  virtual ~AffineTransformEstimator();

  // Estimate the transform between image1 and image 2
  //
  // The return value is .empty() if it fails
  virtual cv::Mat EstimateTransform(const cv::Mat_<double>& image1,
                                    const cv::Mat_<double>& image2) const;

  // Apply a transform to an image
  //
  // Inputs:
  // image - image to transform
  // transform - transform to apply
  // borderFillType - How to fill the pixels outside the frame. See GeometricTransforms.h
  // fillConstant - For BORDER_CONSTANT, what value to fill with
  //
  // Returns the transformed image
  virtual
  cv::Mat_<double> ApplyTransform(
    const cv::Mat_<double>& image,
    const cv::Mat transform,
    geo::BorderFillType borderFillType,
    const double& fillConstant=0.0,
    const cv::Mat_<double>& backgroundImage=cv::Mat_<double>()) const;

  virtual
  cv::Mat_<cv::Vec3d> ApplyTransform(
    const cv::Mat_<cv::Vec3d>& image,
    const cv::Mat transform,
    geo::BorderFillType borderFillType,
    const cv::Vec3d& fillConstant=cv::Vec3d(0,0,0),
    const cv::Mat_<cv::Vec3d>& backgroundImage=cv::Mat_<cv::Vec3d>()) const;

 private:
  int maxIterations_;
  double minPrecision_;
  double imageScaling_;

  
  template <typename T>
  cv::Mat_<T> ApplyTransformImpl(
    const cv::Mat_<T>& image,
    const cv::Mat& transform,
    geo::BorderFillType borderFillType,
    const T& fillConstant=T(),
    const cv::Mat_<T>& backgroundImage=cv::Mat_<T>()) const;
};
  

} // namespace

#endif //  __TRANSFORM_ESTIMATOR_H__
