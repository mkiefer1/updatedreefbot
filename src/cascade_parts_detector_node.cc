#include "cascade_parts_detector/cascade_parts_detector.h"

#include <string>

using namespace std;
using namespace ros;

int main(int argc, char** argv) {
  ros::init(argc, argv, "CascadePartsDetector");

  // Node handle for the parameters for filtering
  ros::NodeHandle localHandle("~");

  string modelFile;
  if (!localHandle.getParam("model_file", modelFile)) {
    ROS_FATAL_STREAM("Could not get the name of the object model file. "
                         "Set the parameter 'model_file' in namespace "
                         << localHandle.getNamespace());
    return 1;
  }

  double thresh;
  localHandle.param("thresh", thresh, 0.0);
  bool do_cascade;
  localHandle.param("do_cascade", do_cascade, false);

  cascade_parts_detector::CascadePartsDetector detector;

  if (!detector.Init(modelFile, thresh, do_cascade)) {
    return 1;
  }

  ros::spin();
}
