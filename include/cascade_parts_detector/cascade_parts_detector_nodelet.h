// Nodelet wrapper for a CascadePartsDetector
//
// Author: Mark Desnoyer (mdesnoyer@gmail.com)

#ifndef __CASCADE_PARTS_DETECTOR_NODELET_H__
#define __CASCADE_PARTS_DETECTOR_NODELET_H__

#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include "cascade_parts_detector.h"

namespace cascade_parts_detector {

class CascadePartsDetectorNodelet : public nodelet::Nodelet {
public:

  virtual void onInit();

private:
  CascadePartsDetector detector_;

};


} // namespace

#endif //__CASCADE_PARTS_DETECTOR_NODELET_H__
