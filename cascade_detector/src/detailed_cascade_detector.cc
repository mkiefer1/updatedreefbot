#include "cascade_detector/detailed_cascade_detector.h"
#include <ext/hash_map>
#include <boost/shared_ptr.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using __gnu_cxx::hash_map;
using namespace cv;
using namespace std;
using boost::shared_ptr;

namespace __gnu_cxx {
template<>
struct hash< cv::Size > : unary_function< cv::Size, size_t > {
  inline size_t operator()(const cv::Size& key) const {
    return (hash<int>()(key.width) * 7823) ^ hash<int>()(key.height);
  }
};
} // end namespace 

namespace cascade_detector {

DetailedCascadeDetector::~DetailedCascadeDetector() {}

bool DetailedCascadeDetector::Init(const std::string& filename) {
  bool retval = load(filename);

  if (this->isOldFormatCascade()) {
    ROS_WARN_STREAM("Using an old format cascade from " << filename);
  }
  return retval;
}

struct PointLists {
  // Points in coordinates of the scaled screen
  shared_ptr<vector<Point> > smallPoints;
  // Indec into the rois for this region
  shared_ptr<vector<unsigned int> > roiIdx; 
};

bool DetailedCascadeDetector::DetectObjects(
  const cv::Mat& image,
  const std::vector<cv::Rect>& roisIn,
  std::vector<cv::Rect>* foundLocations,
  std::vector<int>* scores,
  double* processingTime,
  double* integralTime) {
  ROS_ASSERT(foundLocations);

  if (image.empty()) {
    return true;
  }

  // If the scores are requested, preload them. We'll overwrite them later
  if (scores) {
    scores->clear();
    scores->assign(roisIn.size(), 0);
  }

  Mat greyImage;
  if (image.channels() != 1) {
    cvtColor(image, greyImage, CV_BGR2GRAY);
  } else {
    greyImage = image;
  }

  // For keeping track of the processing time. The measurement is a
  // little tricky because we only want to track the time assuming
  // that the cascade implementation is efficient and doesn't resize
  // the images. Unfortunately, the OpenCV code requires that the
  // images are resized. So, we need to calculate the time to compute
  // the integral image, plus the time to evaluate each window through
  // the cascade.
  ros::Duration processingCount(0);

  // Measure the time to compute the integral image
  CvMat oldSum;
  CvMat oldSqsum;
  CvMat oldTilted;
  if (processingTime) {
    ros::Time startTime = ros::Time::now();
    Mat sum, sqsum;
    integral(greyImage, sum, sqsum, CV_32S);
    processingCount = ros::Time::now() - startTime;
    if (integralTime) {
      *integralTime = processingCount.toSec();
    }
  }

  // First build up a vector of points to evaluate at each model scale
  hash_map<Size, PointLists> evalPoints;
  for(unsigned int i = 0u; i < roisIn.size(); ++i) {
    const cv::Rect& region = roisIn[i];
    double scalingFactor = static_cast<double>(region.height) / 
      GetWindowSize().height;
    Size windowSize(cvRound(greyImage.cols / scalingFactor),
                    cvRound(greyImage.rows / scalingFactor));
    hash_map<Size, PointLists>::iterator pointsPtr =
      evalPoints.find(windowSize);
    if (pointsPtr == evalPoints.end()) {
      PointLists newList;
      newList.smallPoints.reset(new vector<Point>());
      newList.roiIdx.reset(new vector<unsigned int>());
      pointsPtr = evalPoints.insert(pair<Size, PointLists>(windowSize,
                                                           newList)).first;
    }
    pointsPtr->second.smallPoints->push_back(
      Point2i(cvRound(region.x/scalingFactor),
              cvRound(region.y/scalingFactor)));
    pointsPtr->second.roiIdx->push_back(i);
  }

  // Now for each model scale try to detect the object at the points
  for(hash_map<Size, PointLists>::const_iterator i =  evalPoints.begin();
      i != evalPoints.end(); ++i) {
    // Resize the image and get ready to look in it
    Mat smallImage;
    if (i->first.width == greyImage.cols) {
      smallImage = greyImage;
    } else {
      cv::resize(greyImage, smallImage, i->first);
    }

    if (isOldFormatCascade()) {
      Mat sum, sqsum, tilted;
      integral(smallImage, sum, sqsum, tilted, CV_32S);
      oldSum = sum;
      oldSqsum = sqsum;
      oldTilted = tilted;
      cvSetImagesForHaarClassifierCascade(oldCascade,
                                          &oldSum,
                                          &oldSqsum,
                                          &oldTilted,
                                          1.0);
    } else {
      if (!setImage(smallImage)) {
        ROS_ERROR("There was an error setting the image");
        return false;
      }
    }

    // Evaluate the classifier at all the locations in this scale
    Ptr<FeatureEvaluator> evaluator;
    if (!isOldFormatCascade()) {
      evaluator = featureEvaluator->clone();
    }
    ros::Time startTime = ros::Time::now();
    const vector<Point>& smallPoints = *i->second.smallPoints;
    const vector<unsigned int>& roiIdx = *i->second.roiIdx;
    const int nStages = GetNumCascadeStages();
    for (unsigned int j = 0u; j < smallPoints.size(); ++j) {
      int result;
      if (isOldFormatCascade()) {
        result = cvRunHaarClassifierCascade(oldCascade, smallPoints[j], 0);
      } else {
        double garbWeight;
        result = runAt(evaluator, smallPoints[j], garbWeight);
      }
      ROS_ASSERT(result != -GetNumCascadeStages());
      if (result == 1) {
        foundLocations->push_back(roisIn[roiIdx[j]]);
      }
      if (scores) {
        ROS_ASSERT(roiIdx[j] < scores->size());
        (*scores)[roiIdx[j]] = result == 1 ? nStages : -result;
      }
    }
    // Stop the timer
    if (processingTime) {
      processingCount += ros::Time::now() - startTime;
    }
  }

  if (processingTime) {
    *processingTime = processingCount.toSec();
  }

  return true;
}

} // namespace
