// Wraps the OpenCV cascade classifier for a ROS node
//
// Author: Mark Desnoyer (markd@cmu.edu)
// Date: May 2011

#ifndef CASCADE_DETECTOR_H_
#define CASCADE_DETECTOR_H_

#include <string>
#include <opencv2/objdetect/objdetect.hpp>
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "cascade_detector/DetectionArray.h"
#include "cascade_detector/DetectObject.h"

namespace cascade_detector {

class CascadeDetector {
public:
  // Default constructor for an empty classifier.
  CascadeDetector();

  // Constructor that loads a detector from a given file. File is the
  // same format that the cv::CascadeClassifier expects
  CascadeDetector(const std::string& filename, const std::string& objectName);

  // Initialize for connection to ROS.
  void Init(const std::string& imageTopic,
            const std::string& responseTopic,
            const std::string& serviceName);

  // Spins in ROS to handle the messages.
  void Spin() {ros::spin();}

  // Callback for the service
  bool HandleServiceRequest(DetectObject::Request& request,
                            DetectObject::Response& response) {
    return HandleRequestImpl(request.image, &response.detections);
  }

private:
  // The name of this detector
  const std::string objectName_;

  // The OpenCV CascadeClassifier that does all the work
  cv::CascadeClassifier classifier_;

  // The image to process now
  const sensor_msgs::Image* image_;

  // List of detections for this object
  DetectionArray* detections_;

  // The ROS service handler
  ros::ServiceServer service_;

  // The publisher of DetectionArray responses
  ros::Publisher publisher_;

  // The subscriber that listens for Image messages
  ros::Subscriber subscriber_;

  // Parameters for the detection algorithm
  // How much the image size is reduced at each image scale
  double scaleFactor_;
  
  // How many neighbors should each candidate rectangle have to retain it
  int minNeighbors_;
  
  // cvHaarDetectObjects flags
  int flags_;
  
  // The minimum possible object size. Objects smaller than that are ignored
  cv::Size minSize_;

  // Using image_ detect the objects and push them onto detections_
  // This function is the guts
  bool detect();

  // Return the name of the detector
  std::string getName() { return "CascadeDetector"; }

  // Callback that handles an Image and publishes a DetectionArray
  void HandleRequest(const sensor_msgs::Image::ConstPtr& msg);

  bool HandleRequestImpl(const sensor_msgs::Image& image,
                         DetectionArray* response);

  // Disallow evil constructors
  CascadeDetector(const CascadeDetector&);
  void operator=(const CascadeDetector&);

};

} // namespace

#endif // CASCADE_DETECTOR_H_
