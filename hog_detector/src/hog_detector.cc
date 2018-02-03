#include "hog_detector/hog_detector.h"

#include <std_msgs/Duration.h>
#include "cv_utils/CV2ROSConverter.h"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.h"
#include "sensor_msgs/RegionOfInterest.h"

using namespace ros;
using namespace std;
using namespace objdetect_msgs;
using namespace cv;

namespace hog_detector {

HogDetector::~HogDetector() {
}

bool HogDetector::Init(const std::string& objName,
                       const std::string& modelFile, double thresh,
                       bool doTiming, bool useDefaultPeopleDetector,
                       bool doNMS, Size winStride, bool doCache) {

  doTiming_ = doTiming;

  // Load the model
  if (!impl_.InitModel(modelFile, thresh, useDefaultPeopleDetector,
                       doNMS, winStride, doCache)) {
    return false;
  }

  if (useDefaultPeopleDetector) {
    objName_ = "Person";
  } else {
    objName_ = objName;
  }

  if (!InitROS()) {
    return false;
  }
  return true;
}

bool HogDetector::InitROS() {
  ROS_INFO_STREAM("Advertising the service and publisher/subscriber to the ROS master");

  // Now setup the connections for the node
  ros::NodeHandle handle;

  // Setup a ROS service that handles requests
  service_ = handle.advertiseService("detect_object_service",
                                     &HogDetector::HandleServiceRequest,
                                     this);

  // Setup a listener for DetectObject messages
  subscriber_ = handle.subscribe<DetectObject, HogDetector>(
    "detect_object",
    10,
    &HogDetector::HandleRequest,
    this);

  // Setup the publisher to respond to Image messages
  publisher_ = handle.advertise<DetectionArray>(
    "object_detected", 10, false);

  if (doTiming_) {
    timePublisher_ = 
      handle.advertise<std_msgs::Duration>("processing_time",
                                           10,
                                           false);
  }

  return true;
}

void HogDetector::HandleRequest(
  const DetectObject::ConstPtr& msg) {
  DetectionArray::Ptr response(new DetectionArray());
  if (HandleRequestImpl(*msg, response.get())) {
    publisher_.publish(response);
  }
}

bool HogDetector::HandleRequestImpl(const objdetect_msgs::DetectObject& msg,
                                    objdetect_msgs::DetectionArray* response) {
  ROS_ASSERT(response);

  response->header.seq = msg.header.seq;
  response->header.stamp = msg.header.stamp;
  response->header.frame_id = msg.header.frame_id;

  // Get the opencv version of the image
  cv_bridge::CvImageConstPtr cvImagePtr;
  // Makes sure that the converted image stays around as long as this object
  boost::shared_ptr<void const> dummy_object;
  try {
    cvImagePtr = cv_bridge::toCvShare(msg.image,
                                      dummy_object,
                                      sensor_msgs::image_encodings::MONO8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("Could not convert image to OpenCV: %s", e.what());
    return false;
  }
  const Mat& cvImage(cvImagePtr->image);

  // Detect the objects
  vector<Rect> foundLocations;
  vector<double> scores;
  double processingTime;
  if (msg.regions.empty()) {
    impl_.DetectObjects(cvImage, &foundLocations, &scores, &processingTime);
  } else {
    vector<Rect> roisIn;
    cv_utils::ROIs2Rects(msg.regions, &roisIn);
    impl_.DetectObjects(cvImage, roisIn, &foundLocations, &scores,
                        &processingTime);
  }

  // Publish the processing time
  if (doTiming_) {
    timePublisher_.publish(ros::Duration(processingTime));
  }

  // Write the detections to the output message
  for (unsigned int i = 0; i < foundLocations.size(); i++) {
    const Rect& curLocation = foundLocations[i];
    Detection detection;
    detection.header.seq = msg.header.seq;
    detection.header.stamp = msg.header.stamp;
    detection.header.frame_id = msg.header.frame_id;
    detection.label = objName_;
    detection.detector = "HOGDetector";
    detection.score = scores[i];
    Mask mask;
    sensor_msgs::RegionOfInterest roi;
    roi.x_offset = curLocation.x;
    roi.y_offset = curLocation.y;
    roi.width = curLocation.width;
    roi.height = curLocation.height;
    mask.roi = roi;
    detection.mask = mask;
    response->detections.push_back(detection);
  }

  return true;
}

}; // hog_detector
