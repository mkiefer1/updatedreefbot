// Nodelet wrapper for a hog_detector
//
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
// Date: March 2012

#include "hog_detector/hog_detector_nodelet.h"

#include <pluginlib/class_list_macros.h>

using namespace std;

PLUGINLIB_DECLARE_CLASS(hog_detector, HogDetectorNodelet,
                        hog_detector::HogDetectorNodelet,
                        nodelet::Nodelet) 

namespace hog_detector {


void HogDetectorNodelet::onInit() {
  // Node handle for the parameters for filtering
  ros::NodeHandle localHandle("~");

  string objName;
  localHandle.param<string>("obj_name", objName, "");
  string modelFile;
  localHandle.param<string>("model_file", modelFile, "");
  double thresh;
  localHandle.param("thresh", thresh, 0.0);
  bool do_timing;
  localHandle.param("do_timing", do_timing, true);
  bool do_people_detection;
  localHandle.param("do_people_detection", do_people_detection, true);
  bool do_nms;
  localHandle.param("do_nms", do_nms, true);

  if (!detector_.Init(objName, modelFile, thresh, do_timing,
                      do_people_detection, do_nms)) {
    return;
  }
}



} // namespace
