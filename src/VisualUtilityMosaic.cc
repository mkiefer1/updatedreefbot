#include "visual_utility/VisualUtilityMosaic.h"

#include <ros/ros.h>
#include <limits>
#include <opencv2/imgproc/imgproc.hpp>

#include "cv_blobs/BlobResult-Inline.h"

#include "cv_utils/math.h"

using namespace cv;

namespace visual_utility {

VisualUtilityMosaic::~VisualUtilityMosaic() {}

void VisualUtilityMosaic::AddFrame(Mat_<double>& frame,
                                   const Mat* transform) {
  Mat_<double> curFrame = frame;

  // See if we should smmoth the frame using a gaussian smoothing
  if (gaussSigma_ > 0 && !curFrame.empty()) {
    Mat_<double> smoothFrame;
    GaussianBlur(curFrame, smoothFrame, Size(0,0), gaussSigma_, gaussSigma_,
                 BORDER_REPLICATE);
    curFrame = smoothFrame;
  }

  // See if we should smooth the frame using morphological closing
  if (morphCloseSize_ > 0 && !curFrame.empty()) {
    // Stupid morphologyEx function only takes floats....
    Mat_<float> floatP(curFrame);
    morphologyEx(floatP, floatP, MORPH_CLOSE,
                 Mat_<uchar>::ones(morphCloseSize_, morphCloseSize_));
    curFrame = Mat_<double>(floatP);
  }

  if (transform == NULL) {
    if (transformEstimator_.get() == NULL) {
      ROS_WARN("We do not have a transformer and we couldn't calculate it.");
      AddFrameImpl(curFrame, Mat());
    } else {
      // Try to calculate the transform since it wasn't given
      AddFrameImpl(curFrame,
                   transformEstimator_->EstimateTransform(lastFrame_,
                                                          curFrame));
    }
  } else {
    AddFrameImpl(curFrame, *transform);
  }
  lastFrame_ = curFrame;
}

cv::Point2f VisualUtilityMosaic::ToFrameCoords(const cv::Point2f& point) const {
  if (lastFrame_.empty()) {
    return point;
  }
  return cv::Point2f(point.x + lastFrame_.cols / 2.0,
                     point.y + lastFrame_.rows / 2.0);
}

cv::Point2f VisualUtilityMosaic::ToBoresightCoords(const cv::Point2f& point) const {
  if (lastFrame_.empty()) {
    return point;
  }
  return cv::Point2f(point.x - lastFrame_.cols / 2.0,
                     point.y - lastFrame_.rows / 2.0);
}

double VisualUtilityMosaic::GetSumInBox(const cv::Rect_<float>& box) const {
  Point2f lowerLeft(box.x, box.y);
  Point2f lowerRight(box.x+box.width, box.y);
  Point2f upperLeft(box.x, box.y+box.height);
  Point2f upperRight(box.x+box.width, box.y+box.height);
  return GetIntegralValue(upperRight) + GetIntegralValue(lowerLeft) -
    GetIntegralValue(lowerRight) - GetIntegralValue(upperLeft);
}

NullVUMosaic::~NullVUMosaic() {}

void NullVUMosaic::AddFrameImpl(Mat_<double>& frame, const Mat& transform) {
  // Minimal to do because the base class already keeps track of lastFrame_

  integralImage_.reset(NULL);
  mosaicSum_.reset(NULL);
}

const Mat_<double> NullVUMosaic::ExtractRegion(const Rect& roi) const {
  if (lastFrame_.empty()) {
    ROS_WARN("It looks like a frame has not been added to the mosaic yet");
    return lastFrame_;
  }

  if (roi.x >= -lastFrame_.cols / 2.0 && roi.y >= -lastFrame_.rows / 2.0 &&
      roi.x+roi.width <= lastFrame_.cols/2.0 &&
      roi.y+roi.height <= lastFrame_.rows/2.0) {
    return lastFrame_(Rect(roi.x + lastFrame_.cols / 2.0,
                           roi.y + lastFrame_.rows / 2.0,
                           roi.width, roi.height));
  } else {
    ROS_ERROR("Cannot handle locations in the mosaic outside the last frame yet");
    return lastFrame_;
  }
}

double NullVUMosaic::MaximizeFunction(PixelValFunc* func,
                                      Point2f* maxLoc) const {
  double maxVal = -std::numeric_limits<double>::infinity();

  ROS_ASSERT(func);

  if (lastFrame_.empty()) {
    ROS_WARN("It looks like a frame has not been added to the mosaic yet");
    return maxVal;
  }

  int cols = lastFrame_.cols;
  int rows = lastFrame_.rows;
  if (lastFrame_.isContinuous()) {
    cols *= rows;
    rows = 1;
  }
  double val;
  for (int i = 0; i < rows; i++) {
    const double* Mi = lastFrame_[i];
    for(int j = 0; j < cols; j++) {
      val = (*func)(Mi[j]);
      if (val > maxVal) {
        maxVal = val;
        if (maxLoc != NULL) {
          maxLoc->x = j;
          maxLoc->y = i;
        }
      }
    }
  }

  if (maxLoc != NULL) {
    if (lastFrame_.isContinuous()) {
      maxLoc->y = ((int)maxLoc->x) / lastFrame_.cols;
      maxLoc->x = ((int)maxLoc->x) % lastFrame_.cols;
    }
    *maxLoc = ToBoresightCoords(*maxLoc);
  }

  return maxVal;
}

double NullVUMosaic::MaximizeFunction(RegionValFunc* func,
                                      const cv::Size2i& windowSize,
                                      cv::Point2f* maxLoc) const {
  double maxVal = -std::numeric_limits<double>::infinity();

  ROS_ASSERT(func);

  if (lastFrame_.empty()) {
    ROS_ERROR("It looks like a frame has not been added to the mosaic yet");
    return maxVal;
  }

  double val;
  for (int i = 0; i < lastFrame_.rows - windowSize.height; i++) {
    for(int j = 0; j < lastFrame_.cols - windowSize.width; j++) {
      val = (*func)(lastFrame_(Rect_<int>(j, i, windowSize.width,
                                          windowSize.height)));
      if (val > maxVal) {
        maxVal = val;
        if (maxLoc != NULL) {
          maxLoc->x = j + ((float)windowSize.width)/2;
          maxLoc->y = i + ((float)windowSize.height)/2;
        }
      }
    }
  }

  if (maxLoc != NULL) {
    *maxLoc = ToBoresightCoords(*maxLoc);
  }

  return maxVal;
}

double NullVUMosaic::MaximizeFunction(IntegralImageFunc* func,
                                      Point2f* maxLoc) const {
  double maxVal = -std::numeric_limits<double>::infinity();

  ROS_ASSERT(func);

  if (lastFrame_.empty()) {
    ROS_ERROR("It looks like a frame has not been added to the mosaic yet");
    return maxVal;
  }

  double val;
  for (int i = 0; i < lastFrame_.rows; i++) {
    for(int j = 0; j < lastFrame_.cols; j++) {
      val = (*func)(ToBoresightCoords(Point2f(j, i)), *this);
      if (val > maxVal) {
        maxVal = val;
        if (maxLoc != NULL) {
          maxLoc->x = j;
          maxLoc->y = i;
        }
      }
    }
  }

  if (maxLoc != NULL) {
    *maxLoc = ToBoresightCoords(*maxLoc);
  }

  return maxVal;
}

double NullVUMosaic::GetIntegralValue(const cv::Point2f& point) const {
  // Calculate the integral image for this frame if it hasn't been
  // done already
  if (integralImage_.get() == NULL) {
    if (lastFrame_.empty()) {
      return 0.0;
    }
    integralImage_.reset(new Mat_<double>());
    integral(lastFrame_, *integralImage_, CV_64F);
  }

  cv::Point2f coords = ToFrameCoords(point);

  if (coords.x < 0 || coords.y < 0) {
    return 0.0;
  }

  return (*integralImage_)[
    std::min(static_cast<int>(coords.y), integralImage_->rows-1)][
    std::min(static_cast<int>(coords.x), integralImage_->cols-1)];
}

double NullVUMosaic::GetValue(const cv::Point2f& point) const {
  if (lastFrame_.empty()) {
    return 0.0;
  }

  cv::Point2f coords = ToFrameCoords(point);
  if (coords.x < 0 || coords.y < 0 || coords.x >= lastFrame_.cols ||
      coords.y >= lastFrame_.rows) {
    return 0.0;
  }

  return lastFrame_[static_cast<int>(coords.y)][static_cast<int>(coords.x)];
}

double NullVUMosaic::GetSum() const {
  if (mosaicSum_.get() == NULL) {
    mosaicSum_.reset(new double(cv_utils::sum(lastFrame_)));
  }
  return *mosaicSum_;
}

double NullVUMosaic::GetSize() const {
  return lastFrame_.cols * lastFrame_.rows;
}

void NullVUMosaic::FindConnectedComponents(
    PixelValFunc* func,
    double thresh,
    cv_blobs::BlobResult<double>* blobs) const {
  ROS_FATAL("Function not implemented");
}

void NullVUMosaic::FindConnectedComponents(
  double thresh, cv_blobs::BlobResult<double>* blobs) const {

  ROS_ASSERT(blobs != NULL);
  if (lastFrame_.empty()) {
    return;
  }

  blobs->FindBlobs(lastFrame_, thresh);

  // Convert the coordinates for all the blobs found into (0,0) being
  // at the center of the boresight
  *blobs -= Point2i(lastFrame_.cols/2.0, lastFrame_.rows/2.0);
  
}

} // namespace
