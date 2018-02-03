#include "visual_utility/VisualUtilityFilterNodelet.h"

#include <pluginlib/class_list_macros.h>
#include <opencv2/core/core.hpp>
#include <std_msgs/Duration.h>
#include <std_msgs/Float32MultiArray.h>

#include "visual_utility/VisualUtilityEstimator.h"
#include "visual_utility/VisualUtilityMosaic.h"
#include "visual_utility/FrameEstimator.h"
#include "visual_utility/TransformEstimator.h"
#include "visual_utility/cvutils-inl.h"
#include "visual_utility/VisualUtilityROSParams.h"
#include "cv_utils/CV2ROSConverter.h"
#include "cv_utils/math.h"

// The cv bridge changed in versions past cturtle
#ifdef CTURTLE
#include "CvBridge.h"
using sensor_msgs::CvBridge;
#else
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.h"
#endif

using namespace cv;
using namespace boost;
using namespace std;
using sensor_msgs::RegionOfInterest;
using objdetect_msgs::DetectObject;

PLUGINLIB_DECLARE_CLASS(visual_utility, VisualUtilityFilterNodelet,
                        visual_utility::VisualUtilityFilterNodelet,
                        nodelet::Nodelet) 

namespace visual_utility {

VisualUtilityFilterNodelet::~VisualUtilityFilterNodelet() {}

void VisualUtilityFilterNodelet::onInit() {
  InitializeVisualUtilityFilter();

  cameraInfo_.reset(new sensor_msgs::CameraInfo());

  // Setup the connections for the the node
  ros::NodeHandle handle;

  // Setup a listener for Image messages
  imgSubscriber_ = handle.subscribe(
    "image",
    10,
    &VisualUtilityFilterNodelet::HandleImage,
    this);
  cameraInfoSubscriber_ = handle.subscribe(
    "camera_info",
    10,
    &VisualUtilityFilterNodelet::HandleCameraInfo,
    this);

  // Register the publishers
  filteredImagePub_ =
    handle.advertise<DetectObject>("filtered/image",
                                   10, false);
  scorePub_ = handle.advertise<std_msgs::Float32MultiArray>("filtered/scores",
                                                            10, false);
  timePub_ = handle.advertise<std_msgs::Duration>("processing_time",
                                                  10, false);
  debugVuEstimagePub_ = handle.advertise<sensor_msgs::Image>("debug/vuestimate",
                                                             10, false);
  debugMosaicPub_ = handle.advertise<sensor_msgs::Image>("debug/mosaic",
                                                         10, false);
  debugFramePub_ = handle.advertise<sensor_msgs::Image>("debug/frameImage",
                                                        10, false);

  // Advertise the service
  service_ = handle.advertiseService(
    "filter_image_service",
    &VisualUtilityFilterNodelet::HandleImageService, this);
  objdetectService_ = handle.advertiseService(
    "vu_objdetect_service",
    &VisualUtilityFilterNodelet::HandleObjDetectService, this);
}

void VisualUtilityFilterNodelet::InitializeVisualUtilityFilter() {
  // Node handle for the parameters for filtering
  ros::NodeHandle localHandle("~");

  localHandle.param<bool>("visual_debugging", doVisualDebug_, false);

  // Build up the transform estimator to figure out the transform
  // between two frames.
  transformEstimator_.reset(CreateTransformEstimator(localHandle));
  
  // Build up the visual utility estimator
  vuEstimator_.reset(CreateVisualUtilityEstimator(localHandle,
                                                  *transformEstimator_));

  // Build up the visual utiliy mosaic
  vuMosaic_.reset(CreateVisualUtilityMosaic(localHandle,
                                            *transformEstimator_));

  // Build up the frame estimator
  frameEstimator_.reset(CreateFrameEstimator(localHandle));

  filter_.reset(new VisualUtilityFilter(vuEstimator_.get(),
                                        vuMosaic_.get(),
                                        frameEstimator_.get(),
                                        doVisualDebug_));
  
}

bool VisualUtilityFilterNodelet::HandleImageService(
  visual_utility::FilterImage::Request& request,
  visual_utility::FilterImage::Response& response) {

  bool retval = HandleImageImpl(request.image,
                                vector<RegionOfInterest>(),
                                response.regions,
                                response.scores,
                                &response.debug_vu_estimate,
                                &response.debug_mosaic,
                                &response.debug_framing);
  response.image = request.image;

  return retval;
}

bool VisualUtilityFilterNodelet::HandleObjDetectService(
    objdetect_msgs::DetectObjectService::Request& request,
    objdetect_msgs::DetectObjectService::Response& response) {

  vector<RegionOfInterest> roisOut;
  vector<float> scores;

  bool retval = HandleImageImpl(request.request_msg.image,
                                request.request_msg.regions,
                                roisOut,
                                scores,
                                NULL, NULL, NULL); // Debug images

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

void VisualUtilityFilterNodelet::HandleImage(
  const sensor_msgs::ImagePtr& image) {
  sensor_msgs::Image debugVuEstimate;
  sensor_msgs::Image debugMosaic;
  sensor_msgs::Image debugFraming;
  DetectObject outputMsg;
  std_msgs::Float32MultiArray scoreMsg;
  HandleImageImpl(*image, vector<RegionOfInterest>(),
                  outputMsg.regions, scoreMsg.data,
                  &debugVuEstimate, &debugMosaic, &debugFraming);

  // Publish the regions of interest
  outputMsg.image = *image;
  outputMsg.header.stamp = image->header.stamp;
  outputMsg.header.frame_id = image->header.frame_id;
  outputMsg.header.seq = image->header.seq;
  filteredImagePub_.publish(outputMsg);

  // Publish the scores
  scoreMsg.layout.dim[0].size = scoreMsg.data.size();
  scoreMsg.layout.dim[0].stride = scoreMsg.data.size();
  scoreMsg.layout.data_offset = 0;
  scorePub_.publish(scoreMsg);

  if (doVisualDebug_) {
    debugVuEstimagePub_.publish(debugVuEstimate);
    debugMosaicPub_.publish(debugMosaic);
    debugFramePub_.publish(debugFraming);
  }
}

bool VisualUtilityFilterNodelet::HandleImageImpl(
  const sensor_msgs::Image& image,
  const vector<RegionOfInterest>& roisIn,
  vector<RegionOfInterest>& roisOut,
  vector<float>& scores,
  sensor_msgs::Image* debugVuEstimate,
  sensor_msgs::Image* debugMosaic,
  sensor_msgs::Image* debugFraming) {
  ROS_ASSERT(filter_.get() != NULL);

  // First convert the image to OpenCV
#ifdef CTURTLE
  IplImage* iplImage = NULL;
  CvBridge bridge;
  try {
    iplImage = bridge.imgMsgToCv(
      image,
      "bgr8");
  } catch (sensor_msgs::CvBridgeException error) {
    NODELET_ERROR("Could not convert the image to OpenCV");
    return false;
  }
  Mat cvImage(iplImage);
#else
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
#endif // CTURTLE

  // Convert the input rois
  vector<cv::Rect> rects;
  cv_utils::ROIs2Rects(roisIn, &rects);

  std_msgs::Duration timeMsg;
  ros::Time startTime = ros::Time::now();

  // Do the filtering
  vector<Mat> filteredImages;
  filter_->AddImage(cvImage, rects);
  timeMsg.data = ros::Time::now() - startTime;

  timePub_.publish(timeMsg);

  if (filteredImages.empty()) {
    NODELET_WARN("After filtering the image did not find a valid region");
    return true;
  }

  const vector<Rect_<int> >& cvROIs = filter_->GetROI();

  // Now create the output messages
  for (unsigned int i = 0; i < cvROIs.size(); ++i) {
    RegionOfInterest roi;
    roi.x_offset = cvROIs[i].x;
    roi.y_offset = cvROIs[i].y;
    roi.width = cvROIs[i].width;
    roi.height = cvROIs[i].height;
    roisOut.push_back(roi);
  }

  scores = filter_->GetScores();

  if (doVisualDebug_) {
    ToImageMsg(cvutils::NormalizeImage<uchar>(filter_->lastVisualUtility()),
               debugVuEstimate, image.header, "mono8");
    ToImageMsg(cvutils::NormalizeImage<uchar>(filter_->lastMosaic()),
               debugMosaic, image.header, "mono8");
    ToImageMsg(cvutils::NormalizeImage<uchar>(filter_->lastFramingImage()),
               debugFraming, image.header, "mono8");
  }

  return true;

}

void VisualUtilityFilterNodelet::ToImageMsg(const cv::Mat& src,
                                            sensor_msgs::Image* dest,
                                            const roslib::Header& header,
                                            const string& encoding) const {
  if (dest == NULL) return;

#ifdef CTURTLE
  IplImage iplFilteredImage(src);
  sensor_msgs::Image::Ptr curMsg = CvBridge::cvToImgMsg(&iplFilteredImage,
                                                        encoding);
  curMsg->header.stamp = header.stamp;
  curMsg->header.frame_id = header.frame_id;
  curMsg->header.seq = header.seq;
  dest = *curMsg;
#else
  cv_bridge::CvImage bridgeMsg;
  bridgeMsg.header.stamp = header.stamp;
  bridgeMsg.header.frame_id = header.frame_id;
  bridgeMsg.header.seq = header.seq;
  bridgeMsg.encoding = encoding;
  bridgeMsg.image = src;
  bridgeMsg.toImageMsg(*dest);
#endif // CTURTLE
}

} // namespace
