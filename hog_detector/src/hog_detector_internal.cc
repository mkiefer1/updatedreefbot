#include "hog_detector/hog_detector_internal.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <boost/math/special_functions/round.hpp>
#include "cv_utils/nms.h"
#include <boost/shared_ptr.hpp>


#include <ext/hash_map>
using __gnu_cxx::hash_map;

using namespace std;
using namespace boost;
using namespace cv;

namespace __gnu_cxx {
template<>
struct hash< cv::Size > : unary_function< cv::Size, size_t > {
  inline size_t operator()(const cv::Size& key) const {
    return (hash<int>()(key.width) * 7823) ^ hash<int>()(key.height);
  }
};
} // end namespace 

namespace hog_detector {

bool HogDetectorImpl::InitModel(const std::string& modelFile,
                                bool useDefaultPeopleDetector,
                                double thresh,
                                bool doNMS,
                                Size winStride,
                                bool doCache) {
  thresh_ = thresh;
  doNMS_ = doNMS;
  winStride_ = winStride;
  doCache_ = doCache;
  
  // Load the model
  if (useDefaultPeopleDetector) {
    hog_.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
  } else {
    if (!hog_.load(modelFile)) {
      return false;
    }
  }
  return true;
}

struct PointLists {
  // Points in coordinates of the scaled screen
  shared_ptr<vector<Point> > smallPoints;
  // Rectangle in the coordinates of the full screen
  shared_ptr<vector<const Rect*> > largeRects; 
};

// Rounds the number to a given stride location
template <typename T, typename U>
inline U RoundToStride(T val, U stride, U maxLoc) {
  if (stride == 0) {
    return cvRound(val);
  }
  U retval = cvRound(val / stride) * stride;
  if (retval > maxLoc) {
    return retval - stride;
  }
  return retval;
}

bool HogDetectorImpl::DetectObjects(const cv::Mat& image,
                                    const std::vector<cv::Rect>& roisIn,
                                    std::vector<cv::Rect>* foundLocations,
                                    std::vector<double>* scores,
                                    double* processingTime) {
  ROS_ASSERT(foundLocations);
  ROS_ASSERT(scores);

  if (image.empty()) {
    return true;
  }

  bool printedStrideError = false;

  // Start the timer
  ros::WallTime startTime;
  if (processingTime) startTime = ros::WallTime::now();

  // First build up a vector of points to evaluate at each model scale
  hash_map<Size, PointLists> evalPoints;
  for(vector<Rect>::const_iterator regionI = roisIn.begin();
         regionI != roisIn.end(); regionI++) {
    // Check to see if we are at the base size, make sure that the
    // stride is consistent
    if (regionI->height == hog_.winSize.height && !printedStrideError &&
        (regionI->x % winStride_.width != 0 ||
         regionI->y % winStride_.height != 0)) {
      ROS_ERROR("You are asking for a region between the window strides so "
                "the hog values are going to be calulcated for the wrong "
                "location.");
    }

    double scalingFactor = static_cast<double>(regionI->height) / 
      hog_.winSize.height;
    Size windowSize(cvRound(image.cols / scalingFactor),
                    cvRound(image.rows / scalingFactor));
    Size maxLoc = windowSize - hog_.winSize;

    hash_map<Size, PointLists>::iterator pointsPtr =
      evalPoints.find(windowSize);
    if (pointsPtr == evalPoints.end()) {
      PointLists newList;
      newList.smallPoints.reset(new vector<Point>());
      newList.largeRects.reset(new vector<const Rect*>());
      pointsPtr = evalPoints.insert(pair<Size, PointLists>(windowSize,
                                                           newList)).first;
    }

    pointsPtr->second.smallPoints->push_back(
      Point2i(RoundToStride(regionI->x/scalingFactor, winStride_.width,
                            maxLoc.width),
              RoundToStride(regionI->y/scalingFactor, winStride_.height,
                            maxLoc.height)));
    pointsPtr->second.largeRects->push_back(&(*regionI));
  }

  // Now for each model scale try to detect the object at the points
  for(hash_map<Size, PointLists>::const_iterator i =  evalPoints.begin();
      i != evalPoints.end(); ++i) {
    Mat smallImage;
    if (i->first.width == image.cols) {
      smallImage = image;
    } else {
      cv::resize(image, smallImage, i->first);
    }

    vector<Point> curLocations;
    vector<double> curScores;
    Size curWinStride = winStride_;
    if (!doCache_) {
      // We cannot have a win stride if we are not using the cache
      // because internally, the HOG detector is hacked to use the
      // cache if a win stride is specified.
      curWinStride = Size(0,0);
    }
        
    hog_.detect(smallImage, curLocations, curScores,
                thresh_,
                curWinStride, //winStride
                Size(), //padding
                *(i->second.smallPoints));

    if (curLocations.size() != i->second.smallPoints->size()) {
      ROS_ERROR_STREAM("hog_.detect did not not return scores for each "
                       << "point it was given. Asked for: " 
                       << i->second.smallPoints->size() << " points."
                       << " got results for "
                       << curLocations.size() << " points.");
    }

    // Load the detections into data structures that keep track of
    // the global hits.
    for (unsigned int j = 0; j < curLocations.size(); j++) {
      // Record the chosen coordinates in the large frame. This is done
      // assuming that hog_.detect did not change the order of the
      // points to evaluate and provides a value for each point.
      foundLocations->push_back(*(*i->second.largeRects)[j]);

      scores->push_back(curScores[j]);
    }
  }

  if (doNMS_) {
    cv_utils::ApplyGreedyNonMaximalSuppression(foundLocations, scores, 0.5);
  }

  // Stop the timer
  if (processingTime) {
    *processingTime = (ros::WallTime::now() - startTime).toSec();
  }

  return true;
}

bool HogDetectorImpl::DetectObjects(const cv::Mat& image,
                                    std::vector<cv::Rect>* foundLocations,
                                    std::vector<double>* scores,
                                    double* processingTime) {
  ROS_ASSERT(foundLocations);
  ROS_ASSERT(scores);

  if (image.empty()) {
    return true;
  }

  // Start the timer
  ros::WallTime startTime;
  if (processingTime) startTime = ros::WallTime::now();

  hog_.detectMultiScale(image, *foundLocations, *scores,
                        thresh_,
                        Size(), // winStride
                        Size(), // padding
                        1.10, // scale
                        0, // groupThreshold
                        false); // meanshifting

  if (doNMS_) {
    cv_utils::ApplyGreedyNonMaximalSuppression(foundLocations, scores, 0.5);
  }

  // Stop the timer
  if (processingTime) {
    *processingTime = (ros::WallTime::now() - startTime).toSec();
  }

  return true;
}

} // namespace
