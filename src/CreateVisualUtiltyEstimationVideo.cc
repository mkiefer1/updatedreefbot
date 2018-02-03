#include <opencv2/core/core.hpp>
#include <string>
#include <boost/shared_ptr.hpp>

#include "visual_utility/VisualUtilityEstimator.h"
#include "visual_utility/TransformEstimator.h"


using namespace cv;
using namespace boost;
using namespace std;


namespace visual_utility {

VisualUtilityFilterNodelet::~VisualUtilityFilterNodelet() {}

void VisualUtilityFilterNodelet::onInit() {
  InitializeVisualUtilityFilter();

  cameraInfo_.reset(new CameraInfo());

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
  imagePub_ = handle.advertise<sensor_msgs::Image>("filtered/image",
                                                   10, false);
  infoPub_ = handle.advertise<sensor_msgs::CameraInfo>("filtered/camera_info",
                                                       10, false);
  timePub_ = handle.advertise<std_msgs::Duration>("processing_time",
                                                  10, false);

  // Advertise the service
  service_ = handle.advertiseService(
    "filter_image_service",
    &VisualUtilityFilterNodelet::HandleImageService, this);
}

boost::shared_ptr<VisualUtilityEstimator> CreateVisualUtilityEstimator() {
  // Node handle for the parameters for filtering
  ros::NodeHandle localHandle("~");

  // Build up the transform estimator to figure out the transform
  // between two frames.
  int maxIterations;
  double minPrecision;
  double imageScaling;
  localHandle.param<int>("affine_max_iterations", maxIterations, 100);
  localHandle.param<double>("min_affine_precision", minPrecision, 1e-7);
  localHandle.param<double>("affine_scaling_factor", imageScaling, 4.0);
  transformEstimator_.reset(new AffineTransformEstimator(maxIterations,
                                                         minPrecision,
                                                         imageScaling));
  
  // Build up the visual utility estimator
  double paretoThreshold;
  double distDecay;
  string vuEstimatorClass;
  localHandle.param<double>("pareto_thresh", paretoThreshold, 0.03);
  localHandle.param<double>("dist_decay", distDecay, 2.0);
  localHandle.param<string>("vu_estimator", vuEstimatorClass,
                            "LABMotionVUEstimator");
  if (vuEstimatorClass == "LABMotionVUEstimator") {
    vuEstimator_.reset(new LABMotionVUEstimator(*transformEstimator_,
                                                paretoThreshold,
                                                distDecay));
  } else if (vuEstimatorClass == "SpectralSaliency") {
    vuEstimator_.reset(new SpectralSaliency());
  }

  // Build up the visual utiliy mosaic
  int morphCloseSize;
  double gaussSigma;
  localHandle.param<int>("morph_close_size", morphCloseSize, 0);
  localHandle.param<double>("gauss_sigma", gaussSigma, 0.0);
  vuMosaic_.reset(new NullVUMosaic(morphCloseSize, gaussSigma));

  // Build up the frame estimator
  int xFramesize;
  int yFramesize;
  double frameExpansion;
  string frameEstimatorClass;
  localHandle.param<int>("xframesize", xFramesize, 192);
  localHandle.param<int>("yframesize", yFramesize, 108);
  localHandle.param<double>("frame_expansion", frameExpansion, 1.0);
  localHandle.param<string>("frame_estimator", frameEstimatorClass,
                            "MaxPointConstantFramesize");
  if (frameEstimatorClass == "MaxPointConstantFramesize") {
    frameEstimator_.reset(new MaxPointConstantFramesize(
      Size_<int>(xFramesize, yFramesize)));
  } else if (frameEstimatorClass == "RandomPointConstantFramesize") {
    frameEstimator_.reset(new RandomPointConstantFramesize(
      Size_<int>(xFramesize, yFramesize)));
  } else if (frameEstimatorClass == "HighRelativeEntropy") {
    int minFrameArea;
    double minEntropy;
    localHandle.param<int>("min_frame_area", minFrameArea, 200);
    localHandle.param<double>("min_entropy", minEntropy, 0.2);
    frameEstimator_.reset(new HighRelativeEntropy(frameExpansion,
                                                  minFrameArea,
                                                  minEntropy));
  }

  filter_.reset(new VisualUtilityFilter(vuEstimator_.get(),
                                        vuMosaic_.get(),
                                        frameEstimator_.get()));
  
}

bool VisualUtilityFilterNodelet::HandleImageService(
  visual_utility::FilterImage::Request& request,
  visual_utility::FilterImage::Response& response) {

  bool retval = HandleImageImpl(request.image,
                                response.images,
                                response.camera_infos);

  return retval;
}

void VisualUtilityFilterNodelet::HandleImage(
  const sensor_msgs::ImagePtr& image) {
  vector<CameraInfo> cameraInfos;
  vector<sensor_msgs::Image> filteredImageMsgs;
  HandleImageImpl(*image, filteredImageMsgs, cameraInfos);

  for (unsigned int i = 0; i < cameraInfos.size(); ++i) {
    imagePub_.publish(filteredImageMsgs[i]);
    infoPub_.publish(cameraInfos[i]);
  }
}

bool VisualUtilityFilterNodelet::HandleImageImpl(
  const sensor_msgs::Image& image,
  vector<sensor_msgs::Image>& filteredImageMsgs,
  vector<CameraInfo>& cameraInfos) {
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

  std_msgs::Duration timeMsg;
  ros::Time startTime = ros::Time::now();

  // Do the filtering
  vector<Mat> filteredImages;
  filter_->AddImage(cvImage);
  timeMsg.data = ros::Time::now() - startTime;
  filter_->GetFilteredImages(&filteredImages);

  timePub_.publish(timeMsg);

  if (filteredImages.empty()) {
    NODELET_WARN("After filtering the image did not find a valid region");
    return true;
  }

  const vector<Rect_<int> >& rois = filter_->GetROI();

  ROS_ASSERT(rois.size() == filteredImages.size());

  // Now create the output messages
  for (unsigned int i = 0; i < rois.size(); ++i) {
#ifdef CTURTLE
    IplImage iplFilteredImage(filteredImages[i]);
    sensor_msgs::Image::Ptr curMsg = CvBridge::cvToImgMsg(&iplFilteredImage,
                                                          "bgr8");
    curMsg->header.stamp = image.header.stamp;
    curMsg->header.frame_id = image.header.frame_id;
    curMsg->header.seq = image.header.seq;
    filteredImageMsgs.push_back(*curMsg);
#else
    cv_bridge::CvImage bridgeMsg;
    bridgeMsg.header.stamp = image.header.stamp;
    bridgeMsg.header.frame_id = image.header.frame_id;
    bridgeMsg.header.seq = image.header.seq;
    bridgeMsg.encoding = "bgr8";
    bridgeMsg.image = filteredImages[i];
    filteredImageMsgs.push_back(sensor_msgs::Image());
    bridgeMsg.toImageMsg(*(filteredImageMsgs.end()-1));
#endif // CTURTLE
    
    cameraInfos.push_back(*cameraInfo_);
    vector<CameraInfo>::iterator cameraInfo = cameraInfos.end()-1;
    cameraInfo->header.stamp = image.header.stamp;
    cameraInfo->header.frame_id = image.header.frame_id;
    cameraInfo->header.seq = image.header.seq;
    cameraInfo->roi.x_offset += rois[i].x;
    cameraInfo->roi.y_offset += rois[i].y;
    cameraInfo->roi.width = rois[i].width;
    cameraInfo->roi.height = rois[i].height;
  }

  ROS_ASSERT(filteredImageMsgs.size() == cameraInfos.size());

  return true;

}

} // namespace


int main(int argc, char** argv) {
  ros::init(argc, argv, "CreateVisualUtilityVideo");

  if (argc < 3) {
    ROS_FATAL("Usage: CreateVisualUtilityVideo [options] <outputVideo> <listOfFrames.txt>");
    return -1;
  }

  string outputVideo(argv[1]);
  string frameList(argv[2]);

  // Create the visual utility estimator
  

}
