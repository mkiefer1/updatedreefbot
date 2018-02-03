// Copyright 2011 Carnegie Mellon University
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
//
// VisualUtilityFilter.h
//
// A nodelet for running a visual utility filter.
//
// Subscribes to:
// image - a sensor_msgs/Image topic which is the image stream.
//
// Publishes:
// filtered/image - a sensor_msgs/Image topic which is the filtered image steream
// filtered/camera_info - Info about how the image was filtered. Will be spit out with the same timestamp as the corresponding filtered/image

#ifndef __VISUAL_UTILITY_FILTER_NODELET_H__
#define __VISUAL_UTILITY_FILTER_NODELET_H__

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <boost/scoped_ptr.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <objdetect_msgs/DetectObject.h>
#include <objdetect_msgs/DetectObjectService.h>
#include <vector>
#include "VisualUtilityFilter.h"
#include "VisualUtilityEstimator.h"
#include "VisualUtilityMosaic.h"
#include "FrameEstimator.h"
#include "TransformEstimator.h"
#include "visual_utility/FilterImage.h"

namespace visual_utility {

class VisualUtilityFilterNodelet : public nodelet::Nodelet {
public:
  virtual ~VisualUtilityFilterNodelet();

  virtual void onInit();

private:
  ros::Publisher filteredImagePub_;
  ros::Publisher scorePub_;
  ros::Publisher timePub_;
  ros::Subscriber imgSubscriber_;
  ros::Subscriber cameraInfoSubscriber_;
  ros::ServiceServer service_;
  ros::ServiceServer objdetectService_;
  sensor_msgs::CameraInfoPtr cameraInfo_;
  boost::scoped_ptr<VisualUtilityFilter> filter_;
  boost::scoped_ptr<TransformEstimator> transformEstimator_;
  boost::scoped_ptr<VisualUtilityEstimator> vuEstimator_;
  boost::scoped_ptr<VisualUtilityMosaic> vuMosaic_;
  boost::scoped_ptr<FrameEstimator> frameEstimator_;

  //The following are for visual debugging under the hood. If visual
  //debugging is turned on (using the ~/visual_debugging parameter),
  //then we also publish three new image streams: ~/debug/vuestimate,
  //~/debug/mosaic, ~/debug/frameImage
  bool doVisualDebug_;
  ros::Publisher debugVuEstimagePub_;
  ros::Publisher debugMosaicPub_;
  ros::Publisher debugFramePub_;

  bool HandleImageService(visual_utility::FilterImage::Request& request,
                          visual_utility::FilterImage::Response& response);

  bool HandleObjDetectService(
    objdetect_msgs::DetectObjectService::Request& request,
    objdetect_msgs::DetectObjectService::Response& response);

  void HandleImage(const sensor_msgs::ImagePtr& image);

  bool HandleImageImpl(
    const sensor_msgs::Image& image,
    const std::vector<sensor_msgs::RegionOfInterest>& roisIn,
    std::vector<sensor_msgs::RegionOfInterest>& roisOut,
    std::vector<float>& scores,
    sensor_msgs::Image* debugVuEstimate,
    sensor_msgs::Image* debugMosaic,
    sensor_msgs::Image* debugFraming);

  void HandleCameraInfo(const sensor_msgs::CameraInfoPtr& cameraInfo) {
    cameraInfo_ = cameraInfo;
  }

  // Does the initialization and parameter lookup for the visual
  // utility filter
  void InitializeVisualUtilityFilter();

  // Helper function to convert images to image messages
  void ToImageMsg(const cv::Mat& src, sensor_msgs::Image* dest,
                  const roslib::Header& header,
                  const std::string& encoding) const;

};

} // namespace

#endif // __VISUAL_UTILITY_FILTER_NODELET_H__
