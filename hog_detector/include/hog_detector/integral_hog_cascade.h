// A cascade of IntegralHOGDetectors.
//
// Copyright 2012 Mark Desnoyer (mdesnoyer@gmail.com)

#ifndef __HOG_DETECTOR_INTEGRAL_HOG_CASCADE_H__
#define __HOG_DETECTOR_INTEGRAL_HOG_CASCADE_H__

#include <boost/shared_ptr.hpp>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include "hog_detector/integral_hog_detector.h"

namespace hog_detector {

class IntegralHogCascade {
public:
  typedef std::pair<float, IntegralHogDetector::Ptr> Stage;

  IntegralHogCascade() {};
  IntegralHogCascade(const std::string& filename,
                     const cv::Size& winStride);

  // Copy operation. Caller must take ownership
  IntegralHogCascade* copy() const;

  bool load(const std::string& filename);
  void save(const std::string& filename) const;
  bool read(const cv::FileNode& node);
  void write(cv::FileStorage& fs) const;

  // Add a stage to the end of the cascade
  void AddStage(IntegralHogDetector::Ptr& stage, float thresh=0.0) {
    stages_.push_back(Stage(thresh, stage));
  }

  // Computes the score of a window, which is the number of stages past.
  //
  // Inputs:
  // hist - Precomupted integral histogram for this image
  // histSum - Precomputed integral image for the histogram energy
  // roi - Region of interest to compute the score of
  //
  // Outputs:
  // return - The score of the window
  float ComputeScore(const cv_utils::IntegralHistogram<float>& hist,
                     const cv::Mat_<float>& histSum,
                     const cv::Rect& roi) const;

  // From an image, computes the integral histogram needed to compute
  // the HOG descriptor.
  //
  // Inputs:
  // image - Image to compute the oriented gradient histogram of
  //
  // Outputs:
  // histSum - Optional. An integral image of the L1 sum in the histogram
  // returns - The integral histogram. Caller must take ownership.
  cv_utils::IntegralHistogram<float>* ComputeGradientIntegralHistograms(
    const cv::Mat& image,
    cv::Mat_<float>* histSum=NULL) const;

  int GetStageCount() const { return stages_.size(); }

  // To iterate through the stages in the cascade
  std::vector<Stage>::iterator GetStageIterator() { return stages_.begin(); }
  std::vector<Stage>::iterator EndStages() { return stages_.end(); }
  std::vector<Stage>::const_iterator GetStageIterator() const {
    return stages_.begin();
  }
  std::vector<Stage>::const_iterator EndStages() const {
    return stages_.end();
  }

  // To erase a stage
  std::vector<Stage>::iterator EraseStage(std::vector<Stage>::iterator it) {
    return stages_.erase(it);
  }
  

private:
  std::vector<Stage> stages_;

  // Evil constructor
  IntegralHogCascade(const IntegralHogCascade&);
  

};

} // namespace

#endif // __HOG_DETECTOR_INTEGRAL_HOG_CASCADE_H__
