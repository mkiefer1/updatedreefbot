// ROS Node that wraps the OpenCV Haar Cascade Detector for object detection
//
// ROS Input Type: sensor_msgs::Image
// ROS Output Type: cascade_detector::DetectionArray
//
// Author: Mark Desnoyer markd@cmu.edu
// Date: May 2011

#include "ros/ros.h"
#include "cascade_detector/cascade_detector.h"

using namespace std;
using namespace ros;

int main(int argc, char** argv) {
  ros::init(argc, argv, "CascadeDetector");

  // Get the node parameters
  string filename;
  string objectName;
  string imageTopic;
  string responseTopic;
  string serviceName;
  NodeHandle local("~");
  local.getParam("filename", filename);
  local.param<string>("object_name", objectName, "face");
  local.param<string>("image_topic", imageTopic, "detect_object_request");
  local.param<string>("response_topic", responseTopic, "object_detection");
  local.param<string>("service_name", serviceName, "detect_object");

  // Create the node
  cascade_detector::CascadeDetector node(filename, objectName);
  node.Init(imageTopic, responseTopic, serviceName);
  
  node.Spin();
}
  
