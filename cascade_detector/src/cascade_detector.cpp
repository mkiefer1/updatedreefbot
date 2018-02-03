// Wraps the OpenCV cascade classifier for a ROS node
//
// Author: Mark Desnoyer (markd@cmu.edu)
// Date: May 2011

#include "cascade_detector/cascade_detector.h"

#include <boost/shared_ptr.hpp>

#include "cascade_detector/Detection.h"
#include "cascade_detector/Mask.h"

#ifdef CTURTLE
#include "CvBridge.h"
#else
#include "cv_bridge/cv_bridge.h"
#endif

using namespace std;

namespace cascade_detector {

CascadeDetector::CascadeDetector() 
  : objectName_("CascadeDetector") {}

CascadeDetector::CascadeDetector(const string& filename,
                                 const string& objectName)
  : objectName_(objectName), classifier_(filename) {}


void CascadeDetector::Init(const string& imageTopic,
                           const string& responseTopic,
                           const string& serviceName) {
  // Get the parameters for the object detector
  ros::NodeHandle localHandle("~");

  int minSize;

  localHandle.param<double>("scale_factor", scaleFactor_, 1.1);
  localHandle.param<int>("min_neighbors", minNeighbors_, 3);
  localHandle.param<int>("flags", flags_, 0);
  localHandle.param<int>("min_size", minSize, 0);
  minSize_ = cv::Size(minSize, minSize);


  // Now setup the connections for the node
  ros::NodeHandle handle;

  // Setup a ROS service that handles requests
  service_ = handle.advertiseService(serviceName,
                                     &CascadeDetector::HandleServiceRequest,
                                     this);

  // Setup a listener for Image messages
  subscriber_ = handle.subscribe<sensor_msgs::Image, CascadeDetector>(
    imageTopic,
    10,
    &CascadeDetector::HandleRequest,
    this);

  // Setup the publisher to respond to Image messages
  publisher_ = handle.advertise<DetectionArray>(responseTopic, 10, false);
}

void CascadeDetector::HandleRequest(const sensor_msgs::Image::ConstPtr& msg) {
  DetectionArray::Ptr response(new DetectionArray);
  if (HandleRequestImpl(*msg, response.get())) {
    publisher_.publish(response);
  }
}

bool CascadeDetector::HandleRequestImpl(const sensor_msgs::Image& image,
                                        DetectionArray* response) {
  // Copy the time header from the image
  response->header.stamp = image.header.stamp;
  response->header.frame_id = image.header.frame_id;

  image_ = &image;
  detections_ = response;
  bool retval = detect();
  image_ = NULL;
  detections_ = NULL;

  return retval;  
}

bool CascadeDetector::detect() {
  ROS_ASSERT(image_);
  ROS_ASSERT(detections_);

  // First convert the image to OpenCV
#ifdef CTURTLE
  IplImage* iplImage = NULL;
  sensor_msgs::CvBridge bridge;
  try {
    iplImage = bridge.imgMsgToCv(
      *image_,
      "mono8");
  } catch (sensor_msgs::CvBridgeException error) {
    ROS_ERROR("Could not convert the image to OpenCV");
    return false;
  }
  cv::Mat cvImage(iplImage);
#else
  cv_bridge::CvImageConstPtr cvImagePtr;
  // Makes sure that the converted image stays around as long as this object
  boost::shared_ptr<void const> dummy_object;
  try {
    cvImagePtr = cv_bridge::toCvShare(*image_,
                                      dummy_object,
                                      sensor_msgs::image_encodings::MONO8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("Could not convert image to OpenCV: %s", e.what());
    return false;
  }
  cv::Mat& cvImage(cvImagePtr->image);
#endif // CTURTLE

  // Now ask the classifier to find the objects
  vector<cv::Rect> objects;
  classifier_.detectMultiScale(cvImage, objects, scaleFactor_, minNeighbors_,
                               flags_, minSize_);

  // Finally convert the classifier response to the object detection
  // needed by ROS.
  for (vector<cv::Rect>::const_iterator objectI = objects.begin();
       objectI != objects.end();
       ++objectI) {
    Detection detection;
    // Copy the time header from the image
    detection.header.stamp = image_->header.stamp;
    detection.header.frame_id = image_->header.frame_id;

    // Fill in the detection details
    detection.label = objectName_;
    detection.detector = getName();
    detection.score = 1.0;
    detection.mask.roi.x_offset = objectI->x;
    detection.mask.roi.y_offset = objectI->y;
    detection.mask.roi.height = objectI->height;
    detection.mask.roi.width = objectI->width;

    detections_->detections.push_back(detection);
  }
  
  return true;
}

};
