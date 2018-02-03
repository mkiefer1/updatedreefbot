#include "hog_detector/integral_hog_cascade.h"

#include <ros/ros.h>
#include "hog_detector/integral_hog_detector_inl.h"

using namespace cv;
using namespace std;
using namespace boost;

namespace hog_detector {

IntegralHogCascade::IntegralHogCascade(const std::string& filename,
                                       const cv::Size& winStride) {
  if (!load(filename)) {
    ROS_FATAL_STREAM("Could not read cascade from " << filename);
    return;
  }

  for (vector<Stage>::iterator stageI = stages_.begin();
       stageI != stages_.end(); ++stageI) {
    (stageI->second)->SetWinStride(winStride);
  }
}

IntegralHogCascade* IntegralHogCascade::copy() const {
  IntegralHogCascade* retval = new IntegralHogCascade();
  retval->stages_ = stages_;

  return retval;
}

bool IntegralHogCascade::load(const string& filename) {
  FileStorage fs(filename, FileStorage::READ);
  return read(fs.getFirstTopLevelNode());
}
 
void IntegralHogCascade::save(const string& filename) const {
  FileStorage fs(filename, FileStorage::WRITE);
  fs << FileStorage::getDefaultObjectName(filename);
  return write(fs);
}

bool IntegralHogCascade::read(const cv::FileNode& node) {
  if (!node.isMap()) {
    return false;
  }
  FileNode curNode;
  curNode = node["stages_"];
  for (FileNodeIterator i = curNode.begin(); i != curNode.end(); ++i) {
    float thresh;
    (*i)["thresh"] >> thresh;

    IntegralHogDetector::Ptr curStage(new IntegralHogDetector);
    if (!curStage->read((*i)["stage"])) {
      return false;
    }
    AddStage(curStage, thresh);
  }
  return true;
}

void IntegralHogCascade::write(cv::FileStorage& fs) const {
  fs << "{:" "md-integral-hog-cascade";

  fs << "stages_" << "[";
  for (vector<Stage>::const_iterator stageI = stages_.begin();
       stageI != stages_.end(); ++stageI) {
    fs << "{:"
       << "thresh" << stageI->first
       << "stage";
    stageI->second->write(fs);
    fs << "}";
  }
  fs << "]";

  fs << "}";
}

cv_utils::IntegralHistogram<float>*
IntegralHogCascade::ComputeGradientIntegralHistograms(
    const cv::Mat& image,
    cv::Mat_<float>* histSum) const {
  if (stages_.size() > 0 ) {
    return stages_[0].second->ComputeGradientIntegralHistograms(image,
                                                                histSum);
  }
  return NULL;
}

float IntegralHogCascade::ComputeScore(
  const cv_utils::IntegralHistogram<float>& hist,
  const cv::Mat_<float>& histSum,
  const cv::Rect& roi) const {
  for (unsigned int i = 0u; i < stages_.size(); ++i) {
    if (stages_[i].second->ComputeScore(hist, histSum, roi) <
        stages_[i].first) {
      return i;
    }
  }
  return stages_.size();
}

} // namespace
