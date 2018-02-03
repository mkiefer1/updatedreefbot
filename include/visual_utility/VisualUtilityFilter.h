// Copyright 2011 Carnegie Mellon University
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
//
// VisualUtilityFilter.h
//
// An object that can filter an image stream to only extra a frame
// that's defined by the visual utility.
//
// This object is used by adding images to it and then asking for the
// current frame and/or parameters of the sampling. E.g.
//
// cv::Mat rawImage
// cv::Mat filteredImage
// 
// while (camearWorking) {
//   rawImage = GetFrame();
//   filter.AddImage(image);
//   filter.GetFilteredImage(&filteredImage);
//   roi = filter.GetROI();
// }

#ifndef __VISUAL_UTILITY_FILTER_H__
#define __VISUAL_UTILITY_FILTER_H__

#include <opencv2/core/core.hpp>
#include <boost/scoped_ptr.hpp>
#include <vector>
#include "VisualUtilityEstimator.h"
#include "VisualUtilityMosaic.h"
#include "FrameEstimator.h"

namespace visual_utility {

class VisualUtilityFilter {
public:
  // Creates a visual utility filter. Takes ownership of all parameters
  VisualUtilityFilter(VisualUtilityEstimator* vuEstimator,
                      VisualUtilityMosaic* vuMosaic,
                      FrameEstimator* frameEstimator,
                      bool doVisualDebug);

  // Adds a new image to the filter. If rois is not empty, will only
  // evaluate for those regions of interest in the coordinate space of image.
  void AddImage(const cv::Mat& image,
                const std::vector<cv::Rect>& rois);

  // Retreives the most recent frame filtered using the visual utility.
  // The frame could be any size depending on the FrameEstimator used.
  // Multiple areas could also be identified.
  //
  // The filtered frame will be placed in the parameter frame.
  // Note that this will pull from the underlying data input into image,
  // so this function must be called before image is overwritten
  void GetFilteredImages(std::vector<cv::Mat>* frames) const;

  // Retreives the regions of interest of the current filtered frame.
  // The ROI is defined in the coordinates of the raw frame.
  const std::vector<cv::Rect_<int> >& GetROI() const { return roi_; }

  // Retrieves the scores for each of the rois. Order is the same as the rois
  const std::vector<float>& GetScores() const { return scores_; }

  // Functions for retrieving visual debug information
  const cv::Mat_<double>& lastVisualUtility() const {
    return lastVisualUtility_; }
  const cv::Mat_<double>& lastMosaic() const { return lastMosaic_; }
  const cv::Mat& lastFramingImage() const {
    return frameEstimator_->lastDebugImage();
  }
  bool DoVisualDebug() const { return doVisualDebug_; }

private:
  boost::scoped_ptr<VisualUtilityEstimator> vuEstimator_;
  boost::scoped_ptr<VisualUtilityMosaic> vuMosaic_;
  boost::scoped_ptr<FrameEstimator> frameEstimator_;

  std::vector<cv::Rect_<int> > roi_;
  std::vector<float> scores_;
  cv::Mat lastImage_;

  // If this is true, then we populate some intermediate images that
  // help us figure out what's going on under the hood.
  bool doVisualDebug_;
  cv::Mat_<double> lastVisualUtility_;
  cv::Mat_<double> lastMosaic_;  

  // Evil Constructor
  VisualUtilityFilter();

};

} // namespace

#endif // __VISUAL_UTILITY_FILTER_H__
