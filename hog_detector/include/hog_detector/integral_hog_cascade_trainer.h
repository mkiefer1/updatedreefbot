// A cascade of IntegralHOGDetectors.
//
// Copyright 2012 Mark Desnoyer (mdesnoyer@gmail.com)

#ifndef __HOG_DETECTOR_INTEGRAL_HOG_CASCADE_TRAINER_H__
#define __HOG_DETECTOR_INTEGRAL_HOG_CASCADE_TRAINER_H__

#include <ros/ros.h>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <functional>
#include <vector>
#include <string>
#include <queue>
#include <limits>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <gsl/gsl_multimin.h>
#include "hog_detector/integral_hog_cascade.h"
#include "hog_detector/integral_hog_detector.h"
#include "hog_detector/interpolation.h"

namespace hog_detector {

// Class to calculate the various time components of the cascade
class IntegralCascadeTimeCalculator {
public:
  // Creates the calculator based on text files with timing data (in seconds)
  //
  // integralHistTime - single line with the time
  // fillBlockCacheTime - lines of <blockW>,<blockH>,<time>
  // svmEvalTime - lines of <descriptorSize>,<subWinW>,<subWinH>,<time/window>
  // trueHogTime - lines of <nWindows>,<time>
  IntegralCascadeTimeCalculator(const std::string& integralHistTime,
                                const std::string& fillBlockCacheTime,
                                const std::string& svmEvalTime,
                                const std::string& trueHogTime);

  // Time to compute the integral histogram
  double GetIntegralHistTime() const { return integralHistTime_; }

  // Time to fill the block cache for a given block size
  double GetFillCacheTime(const cv::Size& blockSize) const;

  // Time to do the svm evaluation of a single window
  double GetSVMEvalTime(int descriptorSize) const;

  // Time to compute the full HOG detector on N windows
  double GetTrueHogTime(int nWindows) const;

  int GetMaxWindows() const { return maxWindows_; }

  double GetMaxHogTime() const { return GetTrueHogTime(GetMaxWindows()); }

private:
  double integralHistTime_;

  boost::unordered_map<cv::Size, double> cacheFillTime_;

  boost::scoped_ptr<Interpolator> svmTime_;
  boost::scoped_ptr<Interpolator> trueHogTime_;

  int maxWindows_;
};

class IntegralHogCascadeTrainer {
public:
  // Constructor
  //
  // timeCalculator - takes ownership of it
  IntegralHogCascadeTrainer(IntegralCascadeTimeCalculator* timeCalculator,
                            const cv::Size& winSize);

  // Add a candidate IntegralHogDetector.
  void AddCandidateDetector(IntegralHogDetector::Ptr detector) {
    if (detector->winSize() == winSize_) {
      detectorIdx_[detector.get()] = detectors_.size();
      detectors_.push_back(detector);
    } else {
      ROS_WARN_STREAM("The detector has the wrong window size: (" 
                      << winSize_.width << ","
                      << winSize_.height  << ") vs. ("
                      << detector->winSize().width << ","
                      << detector->winSize().height << ")");
    }
  }

  // Trains the cascade detector
  //
  // Inputs:
  // filenames - Filenames of images to train with. Uses the center
  //             winSize window in the image.
  // labels - Labels for all the images. 1 is positive, -1 is negative
  // budget - Processing budget in seconds (average per/frame)
  // missCost - The cost of a miss
  // falsePosCost - The cost of a false positive
  // fracNegSampled - fraction of the number of negative examples we're
  //                  actually training with. So the number of negative 
  //                  samples seen by this class is trueNegs/fracNegSampled
  void Train(const std::vector<std::string>& filenames,
             const std::vector<float>& labels,
             float budget,
             float missCost,
             float falsePosCost);
  // Train using the supermodular-submodular approach
  void TrainSupSub(const std::vector<std::string>& filenames,
                   const std::vector<float>& label,
                   float missCost,
                   float falsePosCost,
                   float timeCostPerError, // time is per seconds
                   float fracNegSampled);

  // Training can also be done by adding one image at a time and then
  // calling TrainWithLoadedData
  void AddImageForTraining(const cv::Mat& image,
                           const std::vector<cv::Rect>& rois,
                           float label);
  void TrainWithLoadedData(float budget,
                           float missCost,
                           float falsePosCost);
  void TrainSupSubWithLoadedData(float missCost,
                                 float falsePosCost,
                                 float timeCostPerError,
                                 float fracNegSampled);

  const IntegralHogCascade* cascade() const { return cascade_.get(); }

private:
  // Structure to hold the score and selected threshold for a given detector
  struct DetectorStats {
    DetectorStats(float _score, float _thresh, unsigned int _detectorIdx)
      : score(_score), thresh(_thresh), detectorIdx(_detectorIdx) {}

    // Sort primarily based on the score
    bool operator<(const DetectorStats& other) const {
      if (fequal(score, other.score)) {
        if (detectorIdx == other.detectorIdx) {
          return thresh < other.thresh;
        }
        return detectorIdx < other.detectorIdx;
      }
      return score < other.score;
    }
    bool operator==(const DetectorStats& other) const {
      return detectorIdx == other.detectorIdx && fequal(score, other.score)
        && fequal(thresh, other.thresh);
    }
    bool operator>(const DetectorStats& other) const {return other < *this;}

    bool fequal(float a, float b,
                float ep=std::numeric_limits<float>::epsilon()*100) const {
      return std::fabs(a-b) < ep;
    }

    float score;
    float thresh;
    unsigned int detectorIdx;
  };
      

  // Number of images being trained on
  int nImages_;

  cv::Size winSize_;

  boost::scoped_ptr<IntegralCascadeTimeCalculator> timeCalculator_;

  // All the possible detectors
  std::vector<IntegralHogDetector::Ptr> detectors_;

  // Scores of the training images for all the detectors. Each row is
  // a different detector and each column is a different training
  // image.
  cv::Mat_<float> scores_;

  // Labels for all the loaded images
  cv::Mat_<float> labels_;

  // Inverse index from the detector to its index in scores_ and labels_
  boost::unordered_map<IntegralHogDetector*, int> detectorIdx_;

  // The trained cascade
  boost::scoped_ptr<IntegralHogCascade> cascade_;

  // The true sum of the number of images given that the negatives are
  // sampled more sparsely.
  double trueImageSum_;

  void LoadImagesForTraining(const std::vector<std::string>& filenames,
                             const std::vector<float>& labels);

  // Trains a cascade using a greedy metric.
  //
  // Inputs:
  // budget - Time budget in seconds,
  // mistCost - Cost of a miss
  // falsePosCost - Cost of a false positive
  // metric - The metric to use
  //
  // Outputs:
  // cascade - The trained cascade
  // returns - Total cost of the cascade
  template <typename T>
  float TrainUsingGreedyMetric(float budget, float missCost,
                               float falsePosCost,
                               T metric,
                               IntegralHogCascade* cascade);

  // Two different metrics for selecting the next best detector.
  //
  // Inputs:
  // missCost - Cost of a miss
  // falsePosCost - Cost of a false positive
  // inputWindows - - Boolean matrix identifying which training windows are
  //                still in play. Will be updated
  // allocatedTime - The time already allocated. Will be updated after this
  //                 selection
  // cacheSizesUsed - Set of cache sizes used. Will be updated after this
  //                  selection.
  //
  // Returns:
  // pair of threshold and index of the selected detector
  std::pair<float, int> SelectBestDetectorIgnoringTime(
    float missCost,
    float falsePosCost,
    cv::Mat& inputWindows,
    float& allocatedTime,
    boost::unordered_set<cv::Size>& cacheSizesUsed);

  std::pair<float, int> SelectBestDetectorPerTime(
    float missCost,
    float falsePosCost,
    cv::Mat& inputWindows,
    float& allocatedTime,
    boost::unordered_set<cv::Size>& cacheSizesUsed);

  // Updates all the tracking data for when a detector is chosen
  void ChooseDetector(int detectorI, float thresh,
                      cv::Mat& inputWindows,
                      float& allocatedTime,
                      boost::unordered_set<cv::Size>& cacheSizesUsed);   


  // Selects the best detector that is under budget
  void SelectBestDetectorUnderBudget(
    float budget, float missCost,
    float falsePosCost,
    const cv::Mat& inputWindows,
    float allocatedTime,
    const boost::unordered_set<cv::Size>& cacheSizesUsed,
    float* thresh,
    int* chosenDetector,
    float* cost);

  // Selects the best detector according to an approximate submodular measure that is tight at cascade.
  // 
  // Inputs:
  //   missCost - Cost of a miss
  //   falsePosCost - Cost of a false positive
  //   timeCostPerError - Conversion factor for 1s of processing time compared
  //                    to an error.
  //   cascade - The best cascade so far. The approximation is tight at
  //               this point.
  //   winPastCascade - Boolean vector specifying the windows that pass the cascade
  //
  // Outputs:
  //  thresh - Threshold of the detector to choose
  //  chosenDetector - Index of the chosen detector
  //  returns - The delta cost for the added detector
  typedef std::priority_queue<DetectorStats, std::vector<DetectorStats>,
                              std::greater<DetectorStats> > DetectorQueue;
  float SelectBestDetectorApprox(
    float missCost,
    float falsePosCost,
    float timeCostPerError,
    float fracNegSampled,
    const IntegralHogCascade& cascade,
    const cv::Mat& winPastApproxCascade,
    DetectorQueue* detectorQueue,
    float* thresh,
    int* chosenDetector);

  // Compute the bayes risk for a given threshold on a set of images
  // whose scores are curScores.
  double ComputeBayesRisk(const cv::Mat& inputWindows,
                          const cv::Mat_<float>& curScores,
                          float thresh,
                          double missCost,
                          double falsePosCost);

  double ComputeProcessingTimeForDetector(
    float fracInputWindows,
    int detectorI,
    const boost::unordered_set<cv::Size>& cacheSizesUsed);

  // For the SupSub proceedure find the cascade that minimizes the
  // approximate cost. The cost is approximate because the D_a and C_M
  // terms are approximated by an upper bound.
  //
  // Inputs:
  // missCost - Cost of a miss
  // falsePosCost - Cost of a false positive, already adjusted with
  //                fracNegSampled
  // timeCostPerError - Conversion factor for 1s of processing time compared
  //                    to an error.
  // bestCascade - The best cascade so far. The approximation is tight at
  //               this point.
  // fracNegSampled - fraction of the number of negative examples we're
  //                  actually training with. So the number of negative 
  //                  samples seen by this class is trueNegs/fracNegSampled
  //
  // Outputs:
  // bestCascade - An update
  // Returns - The true cost of the best cascade (not the approximate cost)
  float MinimizeApproxCost(float missCost, float falsePosCost,
                           float timeCostPerError,
                           float fracNegSampled,
                           boost::scoped_ptr<IntegralHogCascade>* ioCascade);
                
  // Computes the total cost of the cascade
  double ComputeTotalCost(float missCost,
                          float falsePosCost,
                          float timeCostPerError,
                          float fracNegSampled,
                          const IntegralHogCascade& cascade);

  // Initializes the data for computing the approximate submodular cost.
  //
  // Inputs:
  // missCost - Cost of a miss
  // falsePosCost - Cost of a false positive
  // timeCostPerError - Conversion factor for 1s of processing time compared
  //                    to an error.
  // fracNegSampled - fraction of the number of negative examples we're
  //                  actually training with. So the number of negative 
  //                  samples seen by this class is trueNegs/fracNegSampled
  // cascade - The cascade to approximate to
  //
  // Outputs:
  // removeCosts - Map for each stage for the change in cost if it is removed
  // pastCascade - Vector specifying which entries got past the cascade from
  //               rows in scores_
  void InitializeApproxSubCost(
    float missCost,
    float falsePosCost,
    float timeCostPerError,
    float fracNegSampled,
    const IntegralHogCascade& cascade,
    boost::unordered_map<IntegralHogCascade::Stage, float>* removeCosts,
    cv::Mat* pastCascade);

  float CalculateDeltaForRemovingStage(
    float falsePosCost,
    float timeCostPerError,
    float fracNegSampled,
    const IntegralHogCascade& cascade,
    const boost::unordered_map<IntegralHogCascade::Stage, float>& approxCosts,
    const IntegralHogCascade::Stage& stage);

  // Find the best approximate cost for a given detector.
  //
  // missCost, falsePosCost, timeCostPerError, fraceNegSampled - 
  //   relative weights for the costs.
  // scores - score of the detector evaluated on each candidate region
  // usefulPassApproxCascade - usefule regions that pass the approx cascade
  // falsePosWins - False positive windows from the current cascade
  // windowsPassed - Windows that pass the current cascade
  // usefulWindows - Windows that are positives for training
  // notUsefuleWindows - Windows that are negatives for training
  //
  // Returns: Pair of <cost, bestThreshold>
  std::pair<float, float> GetBestApproxCost(
    unsigned int detectorIdx,
    float missCost, 
    float falsePosCost, 
    float timeCostPerError,
    float fracNegSampled,
    double baseHogTime,
    float nPassedApproxCascade,
    const cv::Mat& usefulPassApproxCascade,
    const cv::Mat& falsePosWins,
    const cv::Mat& windowsPassed,
    const cv::Mat& usefulWindows,
    const cv::Mat& notUsefulWindows);
                                            
};

// Class that finds the threshold to minimize the cost of a filter
class DeltaMinimizer {
public:
  DeltaMinimizer();
  ~DeltaMinimizer();

  // Minimizes the cost of adding a current detector
  //
  // Inputs:
  // missCost, falsePosCost, timeCostPerError, fraceNegSampled - 
  //   relative weights for the costs.
  // scores - score of the detector evaluated on each candidate region
  // usefulPassApproxCascade - usefule regions that pass the approx cascade
  // falsePosWins - False positive windows from the current cascade
  // windowsPassed - Windows that pass the current cascade
  // usefulWindows - Windows that are positives for training
  // notUsefuleWindows - Windows that are negatives for training
  // timeCalculator - To calculate the time
  // cpuDelta - cpu cost to run this detector
  float MinimizeCost(float missCost, 
                     float falsePosCost, 
                     float timeCostPerError,
                     float fracNegSampled,
                     int nImages,
                     float trueImageSum,
                     double baseHogTime,
                     const cv::Mat_<float>& scores,
                     const cv::Mat& usefulPassApproxCascade,
                     const cv::Mat& falsePosWins,
                     const cv::Mat& windowsPassed,
                     const cv::Mat& usefulWindows,
                     const cv::Mat& notUsefulWindows,
                     const IntegralCascadeTimeCalculator* timeCalculator,
                     float cpuDelta);

  float bestCost() const { return bestCost_; }
  float bestThresh() const { return bestThresh_; }

  // Evaluates the cost function and its derivative with respect to
  // the number of entries filtered. Public because gsl is old and
  // needs acess to this.
  void EvalWithDeriv(const gsl_vector* nFiltered, void* params,
                     double* val, gsl_vector* derivs) const;
  double EvalCost(const gsl_vector* nFiltered, void* params) const;
  void EvalDeriv(const gsl_vector* nFiltered, void* params,
                 gsl_vector* derivs) const;

private:
  float bestCost_;
  float bestThresh_;

  // Used for the minimization
  gsl_multimin_fdfminimizer* minimizer_;
  gsl_multimin_function_fdf minFunc_;

  // Data used for the calculator
  double cpuDelta_;
  // Map from the number of new windows thresholded to the threshold.
  boost::scoped_ptr<Interpolator> thresholdInterpolator_;
  // Interpolator for the cost of extra misses with a given threshold
  boost::scoped_ptr<Interpolator> missCostInterpolator_;
  // Interpolator for the cost of the false positives relative to the
  // number of new windows thresholded.
  boost::scoped_ptr<Interpolator> falsePosInterpolator_;
  // Interpolator for the HOG time delta
  boost::scoped_ptr<Interpolator> hogTimeInterpolator_;

  void RunMinimization(double minWindowsThreshed, double maxWindowsThreshed,
                       double startPoint, double firstStep,
                       float* bestCost, float* bestThresh);

};

} // namespace

#endif //__HOG_DETECTOR_INTEGRAL_HOG_CASCADE_TRAINER_H__
