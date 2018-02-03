// Internals for the hog detector. Should not be used by a library user.
//
// Author: Mark Desnoyer (mdesnoyer@gmail.com)

#ifndef __HOG_DETECTOR_INTERNAL_H__
#define __HOG_DETECTOR_INTERNAL_H__

#include <ros/ros.h>
#include "objdetect_msgs/DetectObject.h"
#include "objdetect_msgs/DetectionArray.h"
#include "objdetect_msgs/DetectObjectService.h"
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <vector>

namespace hog_detector {

class HogDetectorImpl {
public:
  // Detect objects in an image
  // Inputs:
  // image - image to look for objects in
  // roisIn - rectangles to evaluate
  //
  // Outputs:
  // foundLocations - rectangles where the objects were found
  // scores - score for each of the objects found
  // processingTime - Processing time to find the objects (optional)
  bool DetectObjects(const cv::Mat& image,
                     const std::vector<cv::Rect>& roisIn,
                     std::vector<cv::Rect>* foundLocations,
                     std::vector<double>* scores,
                     double* processingTime);

  // Version of DetectObjects that doesn't have pre-identified regions
  bool DetectObjects(const cv::Mat& image,
                     std::vector<cv::Rect>* foundLocations,
                     std::vector<double>* scores,
                     double* processingTime);

  // Initializes the model for the detector
  bool InitModel(const std::string& modelFile,
                 bool useDefaultPeopleDetector,
                 double thresh,
                 bool doNMS,
                 cv::Size winStride=cv::Size(1, 1),
                 bool doCache=true);

private:
  double thresh_;
  cv::HOGDescriptor hog_;
  bool doNMS_; // Do non-maximal suppression?
  cv::Size winStride_;
  bool doCache_;

};

};

#endif // __HOG_DETECTOR_INTERNAL_H__
