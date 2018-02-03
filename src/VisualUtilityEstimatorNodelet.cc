#include "visual_utility/VisualUtilityEstimatorNodelet.h"
#include <pluginlib/class_list_macros.h>
#include "visual_utility/VisualUtilityROSParams.h"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.h"
#include "cv_utils/CV2ROSConverter.h"

using sensor_msgs::RegionOfInterest;
using namespace cv;
using namespace std;

PLUGINLIB_DECLARE_CLASS(visual_utility, VisualUtilityEstimatorNodelet,
                        visual_utility::VisualUtilityEstimatorNodelet,
                        nodelet::Nodelet) 

namespace visual_utility {

VisualUtilityEstimatorNodelet::~VisualUtilityEstimatorNodelet() {};

void VisualUtilityEstimatorNodelet::onInit() {
  ros::NodeHandle localHandle("~");

  // Build up the transform estimator to figure out the transform
  // between two frames.
  transformEstimator_.reset(CreateTransformEstimator(localHandle));
  
  // Build up the visual utility estimator
  vuEstimator_.reset(CreateVisualUtilityEstimator(localHandle,
                                                  *transformEstimator_));

  // Setup the connections for the the node
  ros::NodeHandle handle;

  // Advertise the service
  objdetectService_ = handle.advertiseService(
    "vu_objdetect_service",
    &VisualUtilityEstimatorNodelet::HandleObjDetectService, this);

  // Advertise the grid service
  gridService_ = handle.advertiseService(
    "vu_objdetect_grid_service",
    &VisualUtilityEstimatorNodelet::HandleObjDetectGridService, this);

}

bool VisualUtilityEstimatorNodelet::HandleObjDetectService(
    objdetect_msgs::DetectObjectService::Request& request,
    objdetect_msgs::DetectObjectService::Response& response) {

  vector<RegionOfInterest> roisOut;
  vector<float> scores;

  bool retval = HandleImageImpl(request.request_msg.image,
                                request.request_msg.regions,
                                roisOut,
                                scores,
                                response.processing_time);

  // Build up the output message
  ROS_ASSERT(roisOut.size() == scores.size());

  for (unsigned int i = 0u; i < roisOut.size(); ++i) {
    objdetect_msgs::Detection curDetection;
    curDetection.score = scores[i];
    curDetection.mask.roi.x_offset = roisOut[i].x_offset;
    curDetection.mask.roi.y_offset = roisOut[i].y_offset;
    curDetection.mask.roi.width = roisOut[i].width;
    curDetection.mask.roi.height = roisOut[i].height;
    response.detections.detections.push_back(curDetection);
  }
  response.detections.header.stamp = request.request_msg.header.stamp;
  response.detections.header.seq = request.request_msg.header.seq;
  response.detections.header.frame_id = request.request_msg.header.frame_id;

  return retval;
}

bool VisualUtilityEstimatorNodelet::HandleImageImpl(
  const sensor_msgs::Image& image,
  const vector<RegionOfInterest>& roisIn,
  vector<RegionOfInterest>& roisOut,
  vector<float>& scores,
  std_msgs::Duration& processingTime) {
  ROS_ASSERT(vuEstimator_.get() != NULL);

  // Convert the input image message to OpenCV
  cv_bridge::CvImageConstPtr cvImagePtr;
  // Makes sure that the converted image stays around as long as this object
  boost::shared_ptr<void const> dummy_object;
  try {
    cvImagePtr = cv_bridge::toCvShare(image,
                                      dummy_object,
                                      sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    NODELET_ERROR("Could not convert image to OpenCV: %s", e.what());
    return false;
  }
  const Mat& cvImage(cvImagePtr->image);

  // Convert the input rois
  vector<cv::Rect> rects;
  cv_utils::ROIs2Rects(roisIn, &rects);

  ros::Time startTime = ros::Time::now();

  // Calculate the visual utility estimation
  vector<VisualUtilityEstimator::ROIScore> vuScores;
  vuEstimator_->CalculateVisualUtility(cvImage,
                                       rects,
                                       image.header.stamp.toSec(),
                                       &vuScores);
  if (vuEstimator_->GetLastRuntime()) {
    processingTime.data = ros::Duration(*vuEstimator_->GetLastRuntime());
  } else {
    processingTime.data = ros::Time::now() - startTime;
  }

  // Now create the output messages
  for (vector<VisualUtilityEstimator::ROIScore>::const_iterator i =
         vuScores.begin(); i != vuScores.end(); ++i) {
    RegionOfInterest roi;
    roi.x_offset = i->second.x;
    roi.y_offset = i->second.y;
    roi.width = i->second.width;
    roi.height = i->second.height;
    roisOut.push_back(roi);
    scores.push_back(i->first);
  }

  return true;

}

bool VisualUtilityEstimatorNodelet::HandleObjDetectGridService(
    objdetect_msgs::DetectObjectGridService::Request& request,
    objdetect_msgs::DetectObjectGridService::Response& response) {
  return HandleGridRequestImpl(request.request_msg,
                               response.scores);
}

bool VisualUtilityEstimatorNodelet::HandleGridRequestImpl(
  const objdetect_msgs::DetectObjectGrid& request,
  objdetect_msgs::DetectGridScores& response) {
  ROS_ASSERT(vuEstimator_.get() != NULL);

  // Copy over pieces to the response
  response.header.stamp = request.header.stamp;
  response.header.seq = request.header.seq;
  response.header.frame_id = request.header.frame_id;
  response.grid.minX = request.grid.minX;
  response.grid.minY = request.grid.minY;
  response.grid.strideX = request.grid.strideX;
  response.grid.strideY = request.grid.strideY;
  response.grid.minW = request.grid.minW;
  response.grid.minH = request.grid.minH;
  response.grid.strideW = request.grid.strideW;
  response.grid.strideH = request.grid.strideH;
  response.grid.fixAspect = request.grid.fixAspect;

  // Pull out the mask and image
  cv_bridge::CvMultiMatConstPtr maskPtr(new cv_bridge::CvMultiMat());
  cv_bridge::CvImageConstPtr cvImagePtr;
  // Makes sure that the converted image stays around as long as this object
  boost::shared_ptr<void const> dummy_object;
  try {
    cvImagePtr = cv_bridge::toCvShare(request.image,
                                      dummy_object,
                                      sensor_msgs::image_encodings::BGR8);
    if (request.mask.sizes.size() > 0) {
      maskPtr = cv_bridge::toCvShare(request.mask,
                                     dummy_object,
                                     sensor_msgs::image_encodings::TYPE_8U);
    }
  } catch (cv_bridge::Exception& e) {
    NODELET_ERROR("Could not convert image to OpenCV: %s", e.what());
    return false;
  }
  const Mat& cvImage(cvImagePtr->image);
  const Mat& mask(maskPtr->mat);

  // Put together the output matrix container
  cv_bridge::CvMultiMat scoreContainer;
  scoreContainer.header.stamp = request.header.stamp;
  scoreContainer.header.seq = request.header.seq;
  scoreContainer.header.frame_id = request.header.frame_id;

  // Calculate the visual utility and time the processing
  ros::Time startTime = ros::Time::now();
  scoreContainer.mat = vuEstimator_->CalculateVisualUtility(
    cvImage,
    request.grid.minX,
    request.grid.minY,
    request.grid.minW,
    request.grid.minH,
    request.grid.strideX,
    request.grid.strideY,
    request.grid.strideW,
    request.grid.strideH,
    request.grid.fixAspect,
    request.header.stamp.toSec(),
    mask);

  if (vuEstimator_->GetLastRuntime()) {
    response.processing_time.data = ros::Duration(
      *vuEstimator_->GetLastRuntime());
  } else {
    response.processing_time.data = ros::Time::now() - startTime;
  }

  // Write the rest of the output
  scoreContainer.toMsg(response.scores);
  maskPtr->toMsg(response.mask);

  return true;
}

} // namespace
