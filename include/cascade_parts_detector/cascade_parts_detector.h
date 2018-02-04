// Wrapper for the MATLAB implementation of the the Discriminatively
// Trained Deformable Part Model object detector
//
// Author: Mark Desnoyer (mdesnoyer@gmail.com)

#ifndef __CASCADE_PARTS_DETECTOR_H__
#define __CASCADE_PARTS_DETECTOR_H__

#include <ros/ros.h>
#include <string.h>
#include "sensor_msgs/Image.h"
#include "cascade_parts_detector/DetectionArray.h"
#include "cascade_parts_detector/DetectObject.h"

// Matlab includes
#include "engine.h"
#include "matrix.h"

namespace cascade_parts_detector {

class CascadePartsDetector {
public:

  // Constructor that loads a detector from a given file into the
  // matlab engine.
  CascadePartsDetector() : matlabEngine_(NULL) {}
  ~CascadePartsDetector();

  // Initialize the node. The model file specified will be loaded into
  // the matlab engine.
  //
  // Inputs:
  // modelFile - Filename of the matlab model to load
  // thresh - Global threshold for detection of a person
  bool Init(const std::string& modelFile, double thresh, bool doCascade,
            bool doTiming=true);

private:
  Engine* matlabEngine_;
  char matlabBuffer_[1024];

  std::string modelFile_;
  std::string thresh_;
  bool doCascade_;
  bool doTiming_;

  // The ROS service handler
  ros::ServiceServer service_;

  // The publisher of DetectionArray responses
  ros::Publisher publisher_;

  // The publisher for the processing time message
  ros::Publisher timePublisher_;

  // The subscriber that listens for Image messages
  ros::Subscriber subscriber_;

  // Inititalizes the connection to ROS
  bool InitROS();

  // Initializes the matlab engine and the variables needed to serve data
  bool InitMatlab();  

  // Shuts down the Matlab engine
  void CloseMatlab();

  // Callback for the service
  bool HandleServiceRequest(DetectObject::Request& request,
                            DetectObject::Response& response) {
    return HandleRequestImpl(request.image, &response.detections);
  }

  // Callback that handles an Image and publishes a DetectionArray
  void HandleRequest(const sensor_msgs::Image::ConstPtr& msg);

  bool HandleRequestImpl(const sensor_msgs::Image& image,
                         DetectionArray* response);

  // Helper function that finds the directory where this package is
  // running from
  std::string FindPackageDir();

  friend class CascadePartsDetectorTest;

};

} // namespace

#endif // __CASCADE_PARTS_DETECTOR_H__
