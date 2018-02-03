// Copyright 2011 Carnegie Mellon University
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
//
// FrameEstimator.h
//
// An object that uses a visual utility mosaic to determine the
// optimal framing to capture some visual utility.
#ifndef __FRAME_ESTIMATOR_H__
#define __FRAME_ESTIMATOR_H__

#include <opencv2/core/core.hpp>
#include <boost/random.hpp>
#include <vector>
#include "VisualUtilityMosaic.h"

namespace visual_utility {

// Structure for the desired frame type. 0,0 is the center of the last
// image, also known as the boresight of the camera.
struct FrameOfInterest {
public:
  FrameOfInterest() {}
  FrameOfInterest(float xCenter_,
                  float yCenter_,
                  float height_,
                  float width_)
    : xCenter(xCenter_), yCenter(yCenter_), height(height_), width(width_){}

  float xCenter;
  float yCenter;
  float height;
  float width;
};

// Abtract base class
class FrameEstimator {
public:
  FrameEstimator(double frameExpansion)
    : frameExpansionFactor_(frameExpansion), displayDebugImages_(false) {}

  virtual ~FrameEstimator();

  // Uses the mosaic to find a number of interesting regions in the mosaic.
  //
  // Inputs:
  // mosaic - Mosaic to search through for interesting regions
  // frames - Output container to store the interesting regions
  virtual void FindInterestingFrames(const VisualUtilityMosaic& mosaic,
                                     std::vector<FrameOfInterest>* frames)=0;

  void SetFullFrameSize(const cv::Size2i& framesize) {
    fullFramesize_ = framesize;
  }

  void DisplayDebugImages(bool v) { displayDebugImages_ = v; }
  void KeepDebugImage(bool v) { keepDebugImage_ = v;}

  const cv::Mat& lastDebugImage() { return lastDebugImage_;}

protected:
  // Defines the framesize of a full image that will be filtered
  cv::Size2i fullFramesize_;

  double frameExpansionFactor_;

  bool displayDebugImages_;

  bool keepDebugImage_;
  cv::Mat lastDebugImage_;

  // Expand the frame by frameExpansionFactor_ in each dimension from
  // the default size that the FrameEstimator finds.
  FrameOfInterest ExpandFrame(const FrameOfInterest& frame);

};

// Looks for a frame around the maximum point and has a constant size
// image extracted.
class MaxPointConstantFramesize : public FrameEstimator {
public:
  MaxPointConstantFramesize(const cv::Size_<int>& framesize);

  virtual ~MaxPointConstantFramesize();

  virtual void FindInterestingFrames(const VisualUtilityMosaic& mosaic,
                                     std::vector<FrameOfInterest>* frames);

private:
  const cv::Size_<int> framesize_;
};

// Creates a frame of a consistent size around a random location
class RandomPointConstantFramesize : public FrameEstimator {
public:
  RandomPointConstantFramesize(const cv::Size_<int>& framesize);

  virtual ~RandomPointConstantFramesize();

  virtual void FindInterestingFrames(const VisualUtilityMosaic& mosaic,
                                     std::vector<FrameOfInterest>* frames);

private:
  const cv::Size_<int> framesize_;

  // Elements for the random number generator
  int seed_;
  boost::mt19937 randomNumberGenerator_;
  boost::uniform_01<> randomRange_;
  boost::variate_generator<boost::mt19937&, boost::uniform_01<> > randNum_;


};

// Dynamically resizes a frame around the maximum value so that it
// maximizes the average visual utility in the frame.
class DynamicResizeAroundMax : public FrameEstimator {
public:
  DynamicResizeAroundMax(double frameExpansion,
                         const cv::Size_<int>& minFramesize)
    : FrameEstimator(frameExpansion), minFramesize_(minFramesize) {}

  virtual ~DynamicResizeAroundMax();

  virtual void FindInterestingFrames(const VisualUtilityMosaic& mosaic,
                                     std::vector<FrameOfInterest>* frames);
private:
  const cv::Size_<int> minFramesize_;
};

// Identifies multiple large regions that have high relative entropy
// compared to a uniform region of visual utility.
class HighRelativeEntropy : public FrameEstimator {
public:
  HighRelativeEntropy(double frameExpansion, int minFrameArea,
                      double minEntropy)
    : FrameEstimator(frameExpansion), minFrameArea_(minFrameArea),
      minEntropy_(minEntropy)
  {}

  virtual ~HighRelativeEntropy();

  virtual void FindInterestingFrames(const VisualUtilityMosaic& mosaic,
                                     std::vector<FrameOfInterest>* frames);

private:
  // Minimum frame area in pixels
  int minFrameArea_;

  // Minimum entropy of a region to be considered
  double minEntropy_;

};

// A frame estimator that returns all the regions in a list whose
// relative entropy are above a threshold
class LocationListWithThreshold : public FrameEstimator {
public:
  // Inputs:
  // thresh - Threshold to be greater than
  // rois - List of locations in the coorindate frame where (0,0) 
  //        is the corner of the most recent image.
  LocationListWithThreshold(double thresh,
                            const std::vector<cv::Rect>& rois,
                            const cv::Size& framesize);

  virtual ~LocationListWithThreshold();

  virtual void FindInterestingFrames(const VisualUtilityMosaic& mosaic,
                                     std::vector<FrameOfInterest>* frames);

private:
  // The regions to sample where (0,0) is the center of the most recent image
  std::vector<FrameOfInterest> regions2Sample_;

  double thresh_;

};

} // namespace

#endif // __FRAME_ESTIMATOR_H__
