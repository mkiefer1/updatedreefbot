// Copyright 2012 Carnegie Mellon University
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
//
// VisualUtilityFilter.h
//
// A nodelet for returning the visual utility filter
//
// Services Provided:
// objdetect_msgs/DetectObjectService

#ifndef __VISUAL_UTILITY_FILTER_NODELET_H__
#define __VISUAL_UTILITY_FILTER_NODELET_H__

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <boost/scoped_ptr.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <std_msgs/Duration.h>
#include <objdetect_msgs/DetectObject.h>
#include <objdetect_msgs/DetectObjectService.h>
#include <objdetect_msgs/DetectObjectGrid.h>
#include <objdetect_msgs/DetectObjectGridService.h>
#include <objdetect_msgs/DetectGridScores.h>
#include <vector>
#include "TransformEstimator.h"
#include "VisualUtilityEstimator.h"

namespace visual_utility {

class VisualUtilityEstimatorNodelet : public nodelet::Nodelet {
public:
  virtual ~VisualUtilityEstimatorNodelet();

  virtual void onInit();

private:
  ros::ServiceServer objdetectService_;
  ros::ServiceServer gridService_;
  boost::scoped_ptr<TransformEstimator> transformEstimator_;
  boost::scoped_ptr<VisualUtilityEstimator> vuEstimator_;

  bool HandleObjDetectService(
    objdetect_msgs::DetectObjectService::Request& request,
    objdetect_msgs::DetectObjectService::Response& response);

  bool HandleObjDetectGridService(
    objdetect_msgs::DetectObjectGridService::Request& request,
    objdetect_msgs::DetectObjectGridService::Response& response);

  bool HandleImageImpl(
    const sensor_msgs::Image& image,
    const std::vector<sensor_msgs::RegionOfInterest>& roisIn,
    std::vector<sensor_msgs::RegionOfInterest>& roisOut,
    std::vector<float>& scores,
    std_msgs::Duration& processingTime);

  bool HandleGridRequestImpl(const objdetect_msgs::DetectObjectGrid& request,
                             objdetect_msgs::DetectGridScores& response);

};

} // namespace

#endif // __VISUAL_UTILITY_FILTER_NODELET_H__
