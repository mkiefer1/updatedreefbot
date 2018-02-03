// Wrapper for the OpenCV HOG detector
//
// Author: Mark Desnoyer (mdesnoyer@gmail.com)

#ifndef __HOG_DETECTOR_H__
#define __HOG_DETECTOR_H__

#include <ros/ros.h>
#include <string.h>
#include "objdetect_msgs/DetectObject.h"
#include "objdetect_msgs/DetectionArray.h"
#include "objdetect_msgs/DetectObjectService.h"
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "hog_detector_internal.h"

namespace hog_detector {

class HogDetector {
public:

  HogDetector() {}
  ~HogDetector();

  // Initialize the node. The model file specified will be loaded into
  // the detector
  //
  // Inputs:
  // objName - Name of the object we are looking for
  // modelFile - Filename of the HOG model to load. 
  // thresh - Global threshold for detection 
  // doTiming - Should the timing of the processing be broadcast on ~processing_time?
  // useDefaultPeopleDetector - Use the default one instead of modelFile. People must be (64,128)
  // useDaimlerPeopleDetector - Use the daimler one instead of modelFile. People must be (48,96)
  bool Init(const std::string& objName,
            const std::string& modelFile, double thresh, 
            bool doTiming=false, bool useDefaultPeopleDetector=false,
            bool doNMS=false,
            cv::Size winStride=cv::Size(),
            bool doCache=true);

private:
  bool doTiming_;
  std::string objName_;
  
  // The implementation
  HogDetectorImpl impl_;

  // The ROS service handler
  ros::ServiceServer service_;

  // The publisher of DetectionArray responses
  ros::Publisher publisher_;

  // The subscriber that listens for DetectObject messages
  ros::Subscriber subscriber_;

  // The publisher for the processing time message
  ros::Publisher timePublisher_;

  // Inititalizes the connection to ROS
  bool InitROS();

  // Callback for the service
  bool HandleServiceRequest(
    objdetect_msgs::DetectObjectService::Request& request,
    objdetect_msgs::DetectObjectService::Response& response) {
    return HandleRequestImpl(request.request_msg, &response.detections);
  }

  // Callback that handles an Image and publishes a DetectionArray
  void HandleRequest(const objdetect_msgs::DetectObject::ConstPtr& msg);

  bool HandleRequestImpl(const objdetect_msgs::DetectObject& msg,
                         objdetect_msgs::DetectionArray* response);

};

} // namespace

#endif // __CASCADE_PARTS_DETECTOR_H__
