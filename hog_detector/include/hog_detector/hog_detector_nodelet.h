// Nodelet wrapper for a hog_detector
//
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
// Date: March 2012

#ifndef __HOG_DETECTOR_NODELET_H__
#define __HOG_DETECTOR_NODELET_H__

#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include "hog_detector.h"

namespace hog_detector {

class HogDetectorNodelet : public nodelet::Nodelet {
public:
  virtual void onInit();

private:
  HogDetector detector_;
};

} // namespace

#endif // __HOG_DETECTOR_NODELET_H__
