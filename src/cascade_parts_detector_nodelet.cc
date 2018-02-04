#include "cascade_parts_detector/cascade_parts_detector_nodelet.h"

#include <pluginlib/class_list_macros.h>
#include <string>

using namespace std;

PLUGINLIB_DECLARE_CLASS(cascade_parts_detector, CascadePartsDetectorNodelet,
                        cascade_parts_detector::CascadePartsDetectorNodelet,
                        nodelet::Nodelet) 

namespace cascade_parts_detector {

void CascadePartsDetectorNodelet::onInit() {
  // Node handle for the parameters for filtering
  ros::NodeHandle localHandle("~");

  string modelFile;
  if (!localHandle.getParam("model_file", modelFile)) {
    NODELET_FATAL_STREAM("Could not get the name of the object model file. "
                         "Set the parameter 'model_file' in namespace "
                         << localHandle.getNamespace());
    return;
  }

  double thresh;
  localHandle.param("thresh", thresh, 0.0);
  bool do_cascade;
  localHandle.param("do_cascade", do_cascade, false);

  if (!detector_.Init(modelFile, thresh, do_cascade)) {
    return;
  }
}

} // namespace
