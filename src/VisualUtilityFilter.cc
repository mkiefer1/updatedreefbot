#include "visual_utility/VisualUtilityFilter.h"

#include <ros/ros.h>
#include <vector>

using namespace cv;
using namespace std;

namespace visual_utility {

inline float bound(float v, float minVal, float maxVal) {
  return v < minVal ? minVal : (v > maxVal ? maxVal : v);
}

inline float max(float valA, float valB) {
  return valA < valB ? valB : valA;
}

VisualUtilityFilter::VisualUtilityFilter(VisualUtilityEstimator* vuEstimator,
                                         VisualUtilityMosaic* vuMosaic,
                                         FrameEstimator* frameEstimator,
                                         bool doVisualDebug)
  : vuEstimator_(vuEstimator), vuMosaic_(vuMosaic),
    frameEstimator_(frameEstimator), roi_(),
    lastImage_(), doVisualDebug_(doVisualDebug) {
  frameEstimator->KeepDebugImage(doVisualDebug);
}

void VisualUtilityFilter::AddImage(const cv::Mat& image,
                                   const vector<Rect>& rois) {
  lastImage_ = image;
  roi_.clear();

  Mat_<double> visualUtility = vuEstimator_->CalculateVisualUtility(image, 0);
  if (doVisualDebug_) {
    lastVisualUtility_ = visualUtility;
  }
  
  vuMosaic_->AddFrame(visualUtility, vuEstimator_->GetLastTransform());

  if (doVisualDebug_) {
    lastMosaic_ = vuMosaic_->ExtractRegion(cv::Rect(-image.cols / 2,
                                                    -image.rows / 2,
                                                    image.cols,
                                                    image.rows));
  }

  // If the rois are specified, force them to be used instead of the
  // loaded frame estimator.
  LocationListWithThreshold overrideFrameEstimator(
    -numeric_limits<double>::infinity(), // threshold
    rois, 
    image.size()); //framesize
  FrameEstimator* frameEstimator = frameEstimator_.get();
  if (!rois.empty()) {
    frameEstimator = &overrideFrameEstimator;
  }

  frameEstimator->SetFullFrameSize(image.size());
  vector<FrameOfInterest> foi;
  frameEstimator->FindInterestingFrames(*vuMosaic_, &foi);

  // TODO(mdesnoyer): Include more complicated camera control

  // Figure out the total size of the frames requested and if they are
  // bigger than the image, just take the whole thing.
  float pixelCount = 0;
  for (vector<FrameOfInterest>::const_iterator i = foi.begin();
       i != foi.end(); ++i) {
    pixelCount += i->width * i->height;
  }
  if (pixelCount > image.rows * image.cols) {
    Rect_<int> roi;
    roi.width = image.cols;
    roi.height = image.rows;
    roi.x = 0;
    roi.y = 0;
    roi_.push_back(roi);
    return;
  }

  for (vector<FrameOfInterest>::const_iterator i = foi.begin();
       i != foi.end(); ++i) {
    Rect_<int> roi;
    roi.width = bound(i->width, 0, image.cols);
    roi.height = bound(i->height, 0, image.rows);
    roi.x = bound(i->xCenter + ((float)image.cols - roi.width)/2, 0,
                  image.cols-((int)roi.width));
    roi.y = bound(i->yCenter + ((float)image.rows - roi.height)/2, 0,
                  image.rows-((int)roi.height));
    roi_.push_back(roi);
  }
}

void VisualUtilityFilter::GetFilteredImages(vector<Mat>* frame) const {
  ROS_ASSERT(frame);

  if (lastImage_.empty()) {
    ROS_ERROR_STREAM("The frame is empty. Did you call AddImage yet?");
    return;
  }

  for (vector<Rect_<int> >::const_iterator i = roi_.begin();
       i != roi_.end(); ++i) {
    frame->push_back(lastImage_(*i));
  }
}

} // namespace
