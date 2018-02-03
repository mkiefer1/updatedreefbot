#include "visual_utility/TimingEstimator.h"

#include <algorithm>
#include <boost/random/uniform_int.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <objdetect_msgs/DetectObjectService.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <cv_utils/CV2ROSConverter.h>
#include <iomanip>

using namespace std;
using namespace cv;
using objdetect_msgs::DetectObjectService;

namespace visual_utility {

double TimingEstimator::GetTiming(int windowCount) {
  // Find the first element not less than windowCount
  vector<SampleType>::const_iterator highElement =
    lower_bound(
      samples_.begin(),
      samples_.end(),
      SampleType(windowCount,
                 0),
      SampleOrdering());

  // Check for corner cases
  if (highElement == samples_.end()) {
    return (highElement-1)->second;
  } else if (highElement == samples_.begin() ||
             highElement->first == windowCount) {
    return highElement->second;
  }

  vector<SampleType>::const_iterator lowElement = highElement - 1;

  // Do the linear interpolation
  return lowElement->second + (windowCount - lowElement->first) *
    (highElement->second - lowElement->second) / 
    (highElement->first - lowElement->first);
}

// A random function operator needed for std::random_shuffle
class RandomGen {
public:
  RandomGen(int maxVal, boost::mt19937* generator) 
    : dist_(0, maxVal), generator_(generator) {}

  std::ptrdiff_t operator()( std::ptrdiff_t arg ) {
    return static_cast<std::ptrdiff_t>(dist_(*generator_));
  }

private:
  boost::uniform_int<> dist_;
  boost::mt19937* generator_;
};

void TimingEstimator::PickNRandomROIs(int N,
                                      const vector<Rect>& inRois,
                                      vector<Rect>* outRois) {
  ROS_ASSERT(outRois);
  outRois->reserve(N);

  // Build a vector of the indices in inRois
  vector<int> idx; idx.reserve(inRois.size());
  for (unsigned int i = 0u; i < inRois.size(); ++i) {
    idx.push_back(i);
  }

  // Shuffle the regions
  RandomGen gen(inRois.size() - 1, &generator_);
  random_shuffle(idx.begin(), idx.end(), gen);

  // Grab the first N
  idx.resize(N);

  // Now sort them
  sort(idx.begin(), idx.end());
  
  // Finally build the output
  for (int i = 0; i < N; ++i) {
    outRois->push_back(inRois[idx[i]]);
  }
}

bool TimingEstimator::LearnVUEstimatorTiming(
  VisualUtilityEstimator* estimator,
  const std::vector<std::string>& images,
  const std::vector<cv::Rect>& rois) {
  ROS_ASSERT(estimator);

  vector<VisualUtilityEstimator::ROIScore> garb;

  // Loop through the images
  for (vector<string>::const_iterator imageFn = images.begin();
       imageFn != images.end(); ++imageFn) {
    Mat image = imread(*imageFn);

    ROS_INFO_STREAM("Timing image: " << *imageFn);

    // Sample from this image for a bunch of different groups of frames
    for (int i = 0; i < sampleCount_+1; ++i) {
      int n = (i) * rois.size() / sampleCount_;
      // Force a sample at n = 1 because that will get the initialization
      if (i == 0) {
        n = 1;
      }

      vector<Rect> subset;
      PickNRandomROIs(n, rois, &subset);
      
      ros::WallTime startTime = ros::WallTime::now();
      estimator->CalculateVisualUtility(image, subset, 1.0, &garb);
      ros::WallDuration measTime = ros::WallTime::now() - startTime;

      if (estimator->GetLastRuntime() != NULL) {
        measTime = ros::WallDuration(*estimator->GetLastRuntime());
      }

      if (samples_.size() <= i) {
        samples_.push_back(SampleType(n, 0.0));
      }
      samples_[i].second += measTime.toSec();
      
      garb.clear();
    }
    
  }

  // Normalize all the timings
  for (vector<SampleType>::iterator i = samples_.begin(); i != samples_.end();
       ++i) {
    i->second /= images.size();
  }

  return true;
}

bool TimingEstimator::LearnObjectDetectorTiming(
  ros::ServiceClient* detector,
  const std::vector<std::string>& images,
  const std::vector<cv::Rect>& rois) {
  ROS_ASSERT(detector);

  // Loop through the images
  for (vector<string>::const_iterator imageFn = images.begin();
       imageFn != images.end(); ++imageFn) {
    Mat image = imread(*imageFn);

    if (image.channels() != 3 || image.depth() != CV_8U) {
      ROS_ERROR("We can only handle 8-bit BGR images here");
      continue;
    }

    // Add the image to the request message
    DetectObjectService::Request request;
    cv_bridge::CvImage bridge;
    bridge.encoding = "bgr8";
    bridge.image = image;
    bridge.toImageMsg(request.request_msg.image);

    // Sample from this image for a bunch of different groups of frames
    for (int i = 0u; i < sampleCount_+1; ++i) {
      int n = (i) * rois.size() / sampleCount_;
      // Force a sample at n = 1 because that will get the initialization
      if (i == 0) {
        n = 1;
      }

      vector<Rect> subset;
      PickNRandomROIs(n,  rois, &subset);

      // Add the ROIs to the request message
      request.request_msg.regions.clear();
      cv_utils::Rects2ROIs(subset, &request.request_msg.regions);

      // Send the request
      DetectObjectService::Response response;
      if (!detector->call(request, response)) {
        ROS_FATAL_STREAM("Failure to call service "
                         << detector->getService());
        return false;
      }
      
      if (samples_.size() <= i) {
        samples_.push_back(SampleType(n, 0.0));
      }
      samples_[i].second += response.processing_time.data.toSec();
    }
    
  }

  // Normalize all the timings
  for (vector<SampleType>::iterator i = samples_.begin(); i != samples_.end();
       ++i) {
    i->second /= images.size();
  }

  return true;
  
}

ostream& TimingEstimator::OutputToStream(std::ostream& stream) {
  stream << "0,0.00" << std::endl;
  for (vector<SampleType>::const_iterator i = samples_.begin();
       i != samples_.end(); ++i) {
    stream << i->first << ',' << setprecision(17) << i->second << std::endl;
  }
  return stream;
}

} // namespace
