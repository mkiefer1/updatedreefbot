#include "visual_utility/TransformEstimator-Inl.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include "visual_utility/GeometricTransforms-Inl.h"

using namespace cv;
using geo::AffineWarp;

namespace visual_utility {

TransformEstimator::~TransformEstimator() {}

AffineTransformEstimator::~AffineTransformEstimator() {}




double MaxArrayValue(const Mat& mat) {
  double retval;
  minMaxLoc(mat, NULL, &retval);
  return retval;
}

Mat AffineTransformEstimator::EstimateTransform(const Mat_<double>& src,
                                                const Mat_<double>& dest) const {
  CV_Assert(src.channels() == 1 && dest.channels() == 1);

  // Resize the image to make the computation faster
  Mat_<double> im1 = src;
  Mat_<double> im2 = dest;
  if (imageScaling_ != 1.0) {
    resize(src, im1, Size(), 1./imageScaling_, 1./imageScaling_,
           INTER_AREA);
    resize(dest, im2, Size(), 1./imageScaling_, 1./imageScaling_,
           INTER_AREA);
  }
      

  // Calculate the x and y gradients of the image
  Mat_<double> kernel = Mat_<double>::zeros(3, 1);
  kernel(0,0) = -0.5;
  kernel(0,2) = 0.5;
  
  Mat_<double> Ix;
  sepFilter2D(im2, Ix, CV_64F, kernel, Mat::ones(1,1,CV_64F));

  Mat_<double> Iy;
  sepFilter2D(im2, Iy, CV_64F, Mat::ones(1,1,CV_64F), kernel);

  // Now build up the A matrix
  long nPixels = im1.cols * im1.rows;
  Mat_<double> A(nPixels, 6);
  int curPixel = 0;
  int rows = Ix.rows;
  int cols = Ix.cols;
  for (int i = 0; i < rows; i++) {
    const double* ixPtr = Ix.ptr<double>(i);
    const double* iyPtr = Iy.ptr<double>(i);
    for (int j = 0; j < cols; j++) {
      double* aPtr = A.ptr<double>(curPixel++);
      aPtr[0] = ixPtr[j] * j;
      aPtr[1] = ixPtr[j] * i;
      aPtr[2] = ixPtr[j];
      aPtr[3] = iyPtr[j] * j;
      aPtr[4] = iyPtr[j] * i;
      aPtr[5] = iyPtr[j];
    }
  }
  Mat invA;
  invert(A, invA, DECOMP_SVD);

  Mat_<double> deltaP = Mat_<double>::eye(3,3);
  Mat_<double> M = Mat_<double>::eye(3, 3);
  Mat_<double> invM(2,3);
  int iterations = 0;
  Mat_<double> warpedImage;
  while (iterations < maxIterations_ &&
         MaxArrayValue(abs(deltaP)) > minPrecision_) {
    invertAffineTransform(M.rowRange(0,2), invM);
    warpedImage = AffineWarp(im1, invM, geo::BORDER_TRANSPARENT, 0.0, im2);

    Mat It = warpedImage - im2;
    It = It.reshape(0, nPixels);

    // Calculate invA * It
    deltaP = -(invA * It);

    // Update M
    Mat_<double> mDiff = Mat_<double>::eye(3, 3);
    mDiff.rowRange(0,2) += deltaP.reshape(2);
    M = mDiff * M;

    iterations++;
    ROS_DEBUG_STREAM("iteration: " << iterations
                     << "\tdeltaP: " << MaxArrayValue(abs(deltaP)) 
                     << "\tdiff:" << MaxArrayValue(abs(It)));
    
  }

  if (iterations == maxIterations_) {
    ROS_WARN("Did not converge");
    return Mat();
  }
  ROS_DEBUG_STREAM("Converged after " << iterations << " iterations.");
  invertAffineTransform(M.rowRange(0,2), invM);

  // If we've scaled the image, we need to rescale the translation
  // parameters so that they are in the full coordinate system.
  invM.colRange(2,3) *= imageScaling_;

  return invM;
}

cv::Mat_<double> AffineTransformEstimator::ApplyTransform(
  const cv::Mat_<double>& image,
  const cv::Mat transform,
  geo::BorderFillType borderFillType,
  const double& fillConstant,
  const cv::Mat_<double>& backgroundImage) const {
  return ApplyTransformImpl(image, transform, borderFillType, fillConstant,
                            backgroundImage);
}

cv::Mat_<Vec3d> AffineTransformEstimator::ApplyTransform(
  const cv::Mat_<Vec3d>& image,
  const cv::Mat transform,
  geo::BorderFillType borderFillType,
  const Vec3d& fillConstant,
  const cv::Mat_<cv::Vec3d>& backgroundImage) const {
  return ApplyTransformImpl(image, transform, borderFillType, fillConstant,
                            backgroundImage);
}

} // namespace
