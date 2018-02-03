// Class that tries to estimate the computational cost for computing a
// specific number of windows. This is created by running some sample
// images through an algorithm, asking for different numbers of
// windows. Then, the timing for a specific number in between the
// sample points is estimated linearily.
//
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
// Date: April, 2012
#ifndef __TIMING_ESTIMATOR_H__
#define __TIMING_ESTIMATOR_H__

#include <ros/ros.h>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <boost/random/mersenne_twister.hpp>
#include "visual_utility/VisualUtilityEstimator.h"
#include <ostream>


namespace visual_utility {

class TimingEstimator {
public:
  // sampleCount - Number of samples to make from 0 to max(windowCount)
  TimingEstimator(int sampleCount, int seed=123456873)
    : sampleCount_(sampleCount), generator_(seed) {}

  // Learn the timing of a visual utility estimator
  bool LearnVUEstimatorTiming(VisualUtilityEstimator* estimator,
                              const std::vector<std::string>& images,
                              const std::vector<cv::Rect>& rois);

  // Learn the timing from a service that accepts
  // objdetect_msgs/DetectObjectService
  bool LearnObjectDetectorTiming(ros::ServiceClient* detector,
                                 const std::vector<std::string>& images,
                                 const std::vector<cv::Rect>& rois);

  // Estimate the timing for a given window count.
  double GetTiming(int windowCount);

  // Prints out a CSV version of the timing estimator where each line
  // is a sample of the form <nWindows>,<timeInSeconds>
  std::ostream& OutputToStream(std::ostream& stream);

private:
  // A type this is a nWindows/runtime pair
  typedef std::pair<int, double> SampleType;

  int sampleCount_;
  boost::mt19937 generator_;

  std::vector<SampleType> samples_;

  struct SampleOrdering {
    bool operator()(const SampleType& a, const SampleType& b) const {
      return a.first < b.first;
    }
  };

  void PickNRandomROIs(int N, const std::vector<cv::Rect>& inRois,
                       std::vector<cv::Rect>* outRois);

};

} // namespace

#endif // __TIMING_ESTIMATOR_H__
