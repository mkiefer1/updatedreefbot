#include "hog_detector/integral_hog_cascade_trainer.h"

#include <ros/ros.h>
#include <fstream>
#include <iostream>
#include <functional>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/functional/hash.hpp>
#include <boost/bind.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include "cv_utils/HashFuncs.h"
#include "cv_utils/IntegralHistogram-Inl.h"

using namespace boost;
using namespace std;
using namespace cv;
using cv_utils::IntegralHistogram;

namespace hog_detector {

IntegralCascadeTimeCalculator::IntegralCascadeTimeCalculator(
  const std::string& integralHistTime,
  const std::string& fillBlockCacheTime,
  const std::string& svmEvalTime,
  const std::string& trueHogTime) {
  // Read the integral hist time
  ifstream integralHistStream(integralHistTime.c_str(), ios_base::in);
  if (!integralHistStream.good()) {
    ROS_ERROR_STREAM("Could not open " << integralHistTime);
    return;
  }
  string curLine;
  integralHistStream >> curLine;
  integralHistTime_ = lexical_cast<double>(curLine);
  integralHistStream.close();

  // Read the time to fill the cache
  ifstream fillCacheStream(fillBlockCacheTime.c_str(), ios_base::in);
  if (!fillCacheStream.good()) {
    ROS_ERROR_STREAM("Could not open " << fillBlockCacheTime);
    return;
  }
  while (fillCacheStream.good() && !fillCacheStream.eof()) {
    fillCacheStream >> curLine;
    vector<string> splitLine;
    boost::split(splitLine, curLine, is_any_of(","));

    if (splitLine.size() != 3) {
      ROS_ERROR_STREAM("Error reading <blockW>,<blockH>,<time>: " << curLine);
      continue;
    }

    cacheFillTime_[cv::Size(lexical_cast<int>(splitLine[0]),
                            lexical_cast<int>(splitLine[1]))] =
      lexical_cast<double>(splitLine[2]);
  }
  fillCacheStream.close();

  // Read the svm evaluation time
  ifstream svmStream(svmEvalTime.c_str(), ios_base::in);
  if (!svmStream.good()) {
    ROS_ERROR_STREAM("Could not open " << svmEvalTime);
    return;
  }
  vector<double> descriptorSizes;
  vector<double> svmTimes;
  while (svmStream.good() && !svmStream.eof()) {
    svmStream >> curLine;
    vector<string> splitLine;
    boost::split(splitLine, curLine, is_any_of(","));

    if (splitLine.size() != 4) {
      ROS_ERROR_STREAM("Error reading <descriptorSize>,<subWinW>,<subWinH>,<time/window>: " << curLine);
      continue;
    }
    descriptorSizes.push_back(lexical_cast<int>(splitLine[0]));
    svmTimes.push_back(lexical_cast<double>(splitLine[3]));
  }
  svmStream.close();
  svmTime_.reset(new SplineInterpolator(descriptorSizes, svmTimes));

  // Read the true hog time
  ifstream trueHogStream(trueHogTime.c_str(), ios_base::in);
  if (!trueHogStream.good()) {
    ROS_ERROR_STREAM("Could not open " << trueHogTime);
    return;
  }
  vector<double> nWindows;
  vector<double> hogTimes;
  while (trueHogStream.good() && !trueHogStream.eof()) {
    trueHogStream >> curLine;
    vector<string> splitLine;
    boost::split(splitLine, curLine, is_any_of(","));

    if (splitLine.size() != 2) {
      ROS_ERROR_STREAM("Error reading <nWindows>,<time>: " << curLine);
      continue;
    }
    nWindows.push_back(lexical_cast<int>(splitLine[0]));
    hogTimes.push_back(lexical_cast<double>(splitLine[1]));
  }
  trueHogStream.close();
  trueHogTime_.reset(new SplineInterpolator(nWindows, hogTimes));
  maxWindows_ = nWindows.back();
}

double IntegralCascadeTimeCalculator::GetFillCacheTime(
  const cv::Size& blockSize) const {
  return cacheFillTime_.at(blockSize);
}

double IntegralCascadeTimeCalculator::GetSVMEvalTime(int descriptorSize) const {
  return (*svmTime_)(descriptorSize);
}

double IntegralCascadeTimeCalculator::GetTrueHogTime(int nWindows) const {
  return (*trueHogTime_)(nWindows);
}

IntegralHogCascadeTrainer::IntegralHogCascadeTrainer(
  IntegralCascadeTimeCalculator* timeCalculator,
  const cv::Size& winSize) 
  : nImages_(0), winSize_(winSize), timeCalculator_(timeCalculator) {}

void IntegralHogCascadeTrainer::LoadImagesForTraining(
  const std::vector<string>& filenames,
  const std::vector<float>& labels) {

  ROS_ASSERT(filenames.size() == labels.size());

  // Load up all the image data
  for (unsigned int i = 0u; i < filenames.size(); ++i) {
    Mat image = imread(filenames[i]);
   
    if (image.empty()) {
      ROS_ERROR_STREAM("Could not open image " << filenames[i] 
                       << " skipping.");
      continue;
    }

    // Grab the middle of the image and add to the training set.
    Rect win((image.cols - winSize_.width) / 2,
             (image.rows - winSize_.height) / 2,
             winSize_.width,
             winSize_.height);
    AddImageForTraining(image, vector<Rect>(1, win), labels[i]);
  }
}

void IntegralHogCascadeTrainer::Train(const std::vector<string>& filenames,
                                      const std::vector<float>& labels,
                                      float budget,
                                      float missCost,
                                      float falsePosCost) {
  LoadImagesForTraining(filenames, labels);

  // Now do the training
  TrainWithLoadedData(budget, missCost, falsePosCost);
}

void IntegralHogCascadeTrainer::TrainSupSub(
  const std::vector<std::string>& filenames,
  const std::vector<float>& labels,
  float missCost,
  float falsePosCost,
  float timeCostPerError,
  float fracNegSampled) {
  LoadImagesForTraining(filenames, labels);

  // Now do the training
  TrainSupSubWithLoadedData(missCost, falsePosCost, timeCostPerError,
                            fracNegSampled);
}

void IntegralHogCascadeTrainer::AddImageForTraining(
  const cv::Mat& image,
  const vector<cv::Rect>& rois,
  float label) {
  ROS_ASSERT(detectors_.size() > 0);

  for (unsigned int i = 0u; i < rois.size(); ++i) {
    labels_.push_back(label);
  }

  // Make the integral histogram for this image
  Mat_<float> histSum;
  scoped_ptr<IntegralHistogram<float> > hist(
    detectors_[0]->ComputeGradientIntegralHistograms(image, &histSum));

  // Build up the scores for this image on each of the detectors. For
  // now, we load scores_ so that each row is an image and each col is
  // the detector.
  Mat_<float> curScores(rois.size(), detectors_.size());
  for (unsigned int i = 0u; i < rois.size(); ++i) {
    for (unsigned int j =0u; j < detectors_.size(); ++j) {
      curScores(i, j) = detectors_[j]->ComputeScore(*hist,
                                                    histSum,
                                                    rois[i]);
    }
  }

  scores_.push_back(curScores);
  nImages_++;
}

void IntegralHogCascadeTrainer::TrainWithLoadedData(float budget,
                                                    float missCost,
                                                    float falsePosCost) {
  // First thing's first, to make things faster, transpose the scores_ matrix
  scores_ = scores_.t();
  labels_ = labels_.t();

  ROS_INFO_STREAM("Starting to train the cascade with "
                  << scores_.cols << " examples");

  // Train a cascade using the greedy heuristic that ignores the
  // processing time
  scoped_ptr<IntegralHogCascade> ignoreTimeCascade(new IntegralHogCascade());
  float ignoreTimeCost = TrainUsingGreedyMetric(
    budget,
    missCost,
    falsePosCost,
    bind(&IntegralHogCascadeTrainer::SelectBestDetectorIgnoringTime, this,
         _1, _2, _3, _4, _5),
    ignoreTimeCascade.get());

  // Train a cascade using the greedy heuristic that gives maximum
  // improvement per unit time.
  scoped_ptr<IntegralHogCascade> perTimeCascade(new IntegralHogCascade());
  float perTimeCost = TrainUsingGreedyMetric(
    budget,
    missCost,
    falsePosCost,
    bind(&IntegralHogCascadeTrainer::SelectBestDetectorPerTime, this,
         _1, _2, _3, _4, _5),
    perTimeCascade.get());

  if (ignoreTimeCost < perTimeCost) {
    cascade_.swap(ignoreTimeCascade);
  } else {
    cascade_.swap(perTimeCascade);
  }
}

void IntegralHogCascadeTrainer::TrainSupSubWithLoadedData(
  float missCost,
  float falsePosCost,
  float timeCostPerError,
  float fracNegSampled) {
  // First thing's first, to make things faster, transpose the scores_ matrix
  scores_ = scores_.t();
  labels_ = labels_.t();

  // Get the true sum of the number of images given that the negatives
  // are sampled more sparsely.
  float nPos = sum((labels_ >= 0) & 0x1)[0];
  float nNeg = sum((labels_ < 0) & 0x1)[0];
  trueImageSum_ = nPos + fracNegSampled * nNeg;

  // Adjust the false positive cost to deal with the fact that there
  // could be less negatives.
  falsePosCost *= fracNegSampled;

  // Get the baseline accuracy cost of no filtering
  double baseAccuracy = nNeg * falsePosCost / nImages_;

  // Adjust the time cost to be relative to equal weighting for
  // accuracy and time when no filter is used.
  timeCostPerError *= timeCalculator_->GetMaxHogTime() / baseAccuracy;

  ROS_INFO_STREAM(
    "Starting to train the cascade using the SupSub proceedure with "
    << scores_.cols << " examples");

  scoped_ptr<IntegralHogCascade> bestCascade(new IntegralHogCascade());
  float bestCost = numeric_limits<float>::infinity();
  for (int nMoves = 0; nMoves < 20; nMoves++) {
    float curCost = MinimizeApproxCost(missCost, falsePosCost,
                                       timeCostPerError, 
                                       fracNegSampled,
                                       &bestCascade);
    if (curCost >= bestCost) {
      ROS_INFO_STREAM("Finished optimization. Best cost is:"
                      << bestCost);
      break;
    }
    bestCost = curCost;
    ROS_INFO_STREAM("Moved to a cascade of size "
                    << bestCascade->GetStageCount()
                    << " with cost " << curCost);
  }

  cascade_.swap(bestCascade);

  // Bound of the D_a term must, when removing a stage, add on the
  // cost to all following stages equal to the windows that were
  // filtered by that stage. This overestimates the extra cost because
  // a later stage may have filtered that window, but that's ok.
}

template<typename T>
float IntegralHogCascadeTrainer::TrainUsingGreedyMetric(
  float budget,
  float missCost,
  float falsePosCost,
  T metric,
  IntegralHogCascade* cascade) {
  ROS_ASSERT(cascade);

  IntegralHogCascade bestCascade;
  IntegralHogCascade curCascade;
  float upperCostBound = numeric_limits<float>::infinity();
  float lowerCostBound = 0;
  float allocatedTime = timeCalculator_->GetIntegralHistTime();
  Mat_<bool> passedWindows = Mat_<bool>::ones(1, scores_.cols);
  unordered_set<Size> cacheSizesUsed;
  while (allocatedTime < budget && lowerCostBound < upperCostBound) {

    // Adjust the upper bound by finding the best cascade stage to add
    // which will minimize the cost and be under the time budget
    float thresh;
    int chosenDetector;
    float cost;
    SelectBestDetectorUnderBudget(budget, missCost, falsePosCost,
                                  passedWindows,
                                  allocatedTime,
                                  cacheSizesUsed,
                                  &thresh,
                                  &chosenDetector,
                                  &cost);
    if (cost < upperCostBound) {
      upperCostBound = cost;
      bestCascade = curCascade;
      bestCascade.AddStage(detectors_[chosenDetector], thresh);
      ROS_INFO_STREAM("Upper cost bound is now: " << upperCostBound);
    }


    // Greedily find the next stage for the cascade
    pair<float, int> metricResult = metric(missCost, falsePosCost,
                                           passedWindows,
                                           allocatedTime,
                                           cacheSizesUsed);
    curCascade.AddStage(detectors_[metricResult.second], metricResult.first);


    lowerCostBound = missCost *
      sum((~passedWindows & (labels_ >= 0)) & 0x1)[0] /
      nImages_;
    ROS_INFO_STREAM("Chose detector " << metricResult.second
                    << " with time allocated " << allocatedTime
                    << " lower bound " << lowerCostBound);
  }

  *cascade = bestCascade;
  return upperCostBound;
}

std::pair<float, int>
IntegralHogCascadeTrainer::SelectBestDetectorIgnoringTime(
  float missCost,
  float falsePosCost,
  cv::Mat& inputWindows,
  float& allocatedTime,
  boost::unordered_set<cv::Size>& cacheSizesUsed) {

  float bestBenefit = -numeric_limits<float>::infinity();
  int chosenDetector = -1;
  float bestThresh = 0;

  const int nInputWindows = sum(inputWindows)[0];
  const float fracInputWindows = ((float)nInputWindows) / scores_.cols;
  const int maxTimeWindows = timeCalculator_->GetMaxWindows();
  const double maxHogTime = timeCalculator_->GetMaxHogTime();
  const double maxBayesRisk = missCost * sum((labels_ >= 0) & 0x1)[0] /
    nImages_;

  for (unsigned int i = 0u; i < detectors_.size(); ++i) {
    const Mat_<float> curScores = scores_.row(i);
    
    // Get the list of unique possible thresholds
    vector<float> validThresh(curScores.begin(), curScores.end());
    std::sort(validThresh.begin(), validThresh.end());
    vector<float>::const_iterator endUnique = 
      std::unique(validThresh.begin(), validThresh.end());

    // Check the thresholds to find the best one
    for(vector<float>::const_iterator threshI = validThresh.begin();
        threshI != endUnique; ++threshI) {
      MatExpr filteredSet = inputWindows & (curScores < *threshI);

      double computingSaved =
        (timeCalculator_->GetTrueHogTime(fracInputWindows * maxTimeWindows) -
         timeCalculator_->GetTrueHogTime(((double)maxTimeWindows) * 
                                         (nInputWindows-sum(filteredSet)[0])/
                                         scores_.cols)) /
         maxHogTime;

      double bayesBenefit = falsePosCost / maxBayesRisk / nImages_ *
        (sum((filteredSet & (labels_ < 0)) & 0x1)[0]);
      

      double curBenefit = computingSaved + bayesBenefit;
      if (curBenefit > bestBenefit) {
        bestBenefit = curBenefit;
        bestThresh = *threshI;
        chosenDetector = i;
      }
    }
  }

  ChooseDetector(chosenDetector, bestThresh,
                 inputWindows,
                 allocatedTime,
                 cacheSizesUsed);

  return pair<float, int>(bestThresh, chosenDetector);
}

std::pair<float, int> IntegralHogCascadeTrainer::SelectBestDetectorPerTime(
    float missCost,
    float falsePosCost,
    cv::Mat& inputWindows,
    float& allocatedTime,
    boost::unordered_set<cv::Size>& cacheSizesUsed) {
  const int nInputWindows = sum(inputWindows)[0];
  const float fracInputWindows = ((float)nInputWindows) / scores_.cols;
  const int maxTimeWindows = timeCalculator_->GetMaxWindows();
  const double maxHogTime = timeCalculator_->GetMaxHogTime();
  const double maxBayesRisk = missCost * sum((labels_ >= 0) & 0x1)[0] /
    nImages_;

  float bestBenefit = -numeric_limits<float>::infinity();
  int chosenDetector = -1;
  float bestThresh = 0;

  for (unsigned int i = 0u; i < detectors_.size(); ++i) {
    const Mat_<float> curScores = scores_.row(i);

    // Get the list of unique possible thresholds
    vector<float> validThresh(curScores.begin(), curScores.end());
    std::sort(validThresh.begin(), validThresh.end());
    vector<float>::const_iterator endUnique = 
      std::unique(validThresh.begin(), validThresh.end());

    // Check the thresholds to find the best one by reduction in Bayes
    // Risk per computation time.
    for(vector<float>::const_iterator threshI = validThresh.begin();
        threshI != endUnique; ++threshI) {
      
      MatExpr filteredSet = inputWindows & (curScores < *threshI);

      // Calculate the changes in computing time weighted by the time
      // without any filters.
      double computingSaved =
        (timeCalculator_->GetTrueHogTime(fracInputWindows * maxTimeWindows) -
         timeCalculator_->GetTrueHogTime(((double)maxTimeWindows) * 
                                         (nInputWindows-sum(filteredSet)[0])/
                                         scores_.cols)) /
         maxHogTime;

      double computingTimeCost =
        ComputeProcessingTimeForDetector(fracInputWindows, i, cacheSizesUsed)/
        maxHogTime;


      // Calculate the changes in the bayes risk between having
      // this detector and not.
      double bayesBenefit = falsePosCost / maxBayesRisk / nImages_ *
        (sum((filteredSet & (labels_ < 0)) & 0x1)[0]);
      double bayesCost = missCost / maxBayesRisk / nImages_ * 
        (sum((filteredSet & (labels_ >= 0)) & 0x1)[0]);

      // Calculate the benefit per cost
      double curBenefit = (computingSaved + bayesBenefit) /
        (computingTimeCost + bayesCost);

      if (curBenefit > bestBenefit) {
        bestBenefit = bestBenefit;
        bestThresh = *threshI;
        chosenDetector = i;
      }
    }
  }

  ChooseDetector(chosenDetector, bestThresh,
                 inputWindows,
                 allocatedTime,
                 cacheSizesUsed);

  return pair<float, int>(bestThresh, chosenDetector);
}

void IntegralHogCascadeTrainer::ChooseDetector(
  int detectorI, float thresh,
  cv::Mat& inputWindows,
  float& allocatedTime,
  boost::unordered_set<cv::Size>& cacheSizesUsed) {
  if (detectorI < 0) {
    ROS_WARN_STREAM("There was no detector to choose");
    return;
  }

  float fracInputWindows = ((float)sum(inputWindows)[0]) / scores_.cols;

  // Update which windows are being passed on
  inputWindows &= (0x1 & (scores_.row(detectorI) >= thresh));

  // Allocate the time to compute the descriptor on all the windows
  // that were passed to this detector and to fill the cache
  allocatedTime += ComputeProcessingTimeForDetector(fracInputWindows,
                                                    detectorI,
                                                    cacheSizesUsed);

  // Update the list of block sizes used
  cacheSizesUsed.insert(detectors_[detectorI]->generator().blockSize());
}

double IntegralHogCascadeTrainer::ComputeProcessingTimeForDetector(
  float fracInputWindows,
  int detectorI,
  const boost::unordered_set<cv::Size>& cacheSizesUsed) {
  // The time to do the svm evaluations
  double svmTime = fracInputWindows * timeCalculator_->GetMaxWindows() *
    timeCalculator_->GetSVMEvalTime(
      detectors_[detectorI]->generator().GetDescriptorSize());

  // The time for filling a new cache
  Size blockSize = detectors_[detectorI]->generator().blockSize();
  double cacheTime = 0;
  if (cacheSizesUsed.find(blockSize) ==  cacheSizesUsed.end()) {
    cacheTime = timeCalculator_->GetFillCacheTime(blockSize);
  }

  return svmTime + cacheTime;
}

// Predicate for x >= val 
template <typename T>
struct greater_eq_const {
  greater_eq_const(const T& val) : val_(val) {}
  bool operator()(const T& x) { return x >= val_; }
  const T& val_;
};

void IntegralHogCascadeTrainer::SelectBestDetectorUnderBudget(
    float budget, float missCost,
    float falsePosCost,
    const cv::Mat& inputWindows,
    float allocatedTime,
    const boost::unordered_set<cv::Size>& cacheSizesUsed,
    float* thresh,
    int* chosenDetector,
    float* cost) {
  const float maxTimeWindows = timeCalculator_->GetMaxWindows();
  const float fracInputWindows = ((float)sum(inputWindows)[0]) / scores_.cols;

  *cost = numeric_limits<float>::infinity();
  for (unsigned int i = 0u; i < detectors_.size(); ++i) {
    const Mat_<float> curScores = scores_.row(i);

    double budgetLeft = budget - allocatedTime;
    budgetLeft -= ComputeProcessingTimeForDetector(fracInputWindows,
                                                   i,
                                                   cacheSizesUsed);

    if (budgetLeft < 0) {
      // No budget left for this detector
      continue;
    }

    // In this detector, find the best threshold that keeps us under
    // budget. To do this, we first find the minimum threshold where
    // we stay under budget.
    double minThresh;
    double maxThresh;
    minMaxLoc(curScores, &minThresh, &maxThresh);
    double leftThresh = maxThresh;
    double lastBudgetUsed = -1;
    double curBudgetUsed = 0;
    while(lastBudgetUsed != curBudgetUsed) {
      lastBudgetUsed = curBudgetUsed;

      double curThresh = (minThresh + leftThresh) / 2;

      int nWindowsPastThresh = sum(
        (inputWindows & (curScores >= curThresh)) & 0x1)[0];

      curBudgetUsed = timeCalculator_->GetTrueHogTime(maxTimeWindows *
                                                      nWindowsPastThresh /
                                                      scores_.cols);

      if (curBudgetUsed > budgetLeft) {
        // Threshold needs to tighten
        minThresh = curThresh;
      } else {
        // Threshold can open up
        leftThresh = curThresh;
      }
    }

    // Now that we have the minimum threshold to stay in budget, find
    // the one that minimizes the cost. We first filter out all those
    // possible thresholds that are not unique and we cannot use
    vector<float> validThresh(curScores.begin(), curScores.end());
    std::sort(validThresh.begin(), validThresh.end(), std::greater<float>());
    vector<float>::const_iterator endUnique = 
      std::unique(validThresh.begin(), validThresh.end());
                       
    // Finally check the valid thresholds.
    for(vector<float>::const_iterator threshI = validThresh.begin();
        threshI != endUnique; ++threshI) {
      if (*threshI < minThresh) {
        break;
      }

      double curCost = ComputeBayesRisk(inputWindows, curScores, *threshI,
                                        missCost, falsePosCost);
      if (curCost < *cost) {
        *cost = curCost;
        *thresh = *threshI;
        *chosenDetector = i;
      }
    }
  }
}

double IntegralHogCascadeTrainer::ComputeBayesRisk(
  const Mat& inputWindows,
  const Mat_<float>& curScores,
  float thresh,
  double missCost,
  double falsePosCost) {
  // Create binary linst of those windows that got to this filter and
  // pass the threshold.
  MatExpr passFilter = inputWindows & (curScores >= thresh);

  // Compute the Bayes Risk
  return (missCost * sum((~passFilter & (labels_ >= 0)) & 0x1)[0] +
          falsePosCost * sum((passFilter & (labels_ < 0)) & 0x1)[0]) /
    nImages_;
}

float IntegralHogCascadeTrainer::MinimizeApproxCost(
  float missCost,
  float falsePosCost,
  float timeCostPerError,
  float fracNegSampled,
  boost::scoped_ptr<IntegralHogCascade>* ioCascade) {
  ROS_ASSERT(ioCascade);

  DetectorQueue detectorQueue;
  bool queueInitialized = false;

  // Initialize for the approximation, which only depends on the
  // entries in the current cascade.
  unordered_map<IntegralHogCascade::Stage, float> approxSubCost;
  Mat winPastCascade;
  InitializeApproxSubCost(missCost, falsePosCost, timeCostPerError,
                          fracNegSampled, **ioCascade, &approxSubCost,
                          &winPastCascade);
  

  scoped_ptr<IntegralHogCascade> bestCascade((*ioCascade)->copy());
  bool foundBetterCascade = true;
  while (foundBetterCascade && !(queueInitialized && detectorQueue.empty())) {
    foundBetterCascade = false;
    queueInitialized = true;

    // Find the maximum improvement in the cascade by removing a stage
    float bestSmallerDelta = 0;
    vector<IntegralHogCascade::Stage>::iterator stage2Remove = 
      bestCascade->EndStages();
    for (vector<IntegralHogCascade::Stage>::iterator stageI = 
           bestCascade->GetStageIterator();
         stageI != bestCascade->EndStages();
         ++stageI) {
      float curDelta = CalculateDeltaForRemovingStage(falsePosCost,
                                                      timeCostPerError,
                                                      fracNegSampled,
                                                      *bestCascade,
                                                      approxSubCost,
                                                      *stageI);
      if (curDelta < bestSmallerDelta) {
        bestSmallerDelta = curDelta;
        stage2Remove = stageI;
      }
    }

    // Find the maximum improvement in the cascade by adding a stage
    float chosenThresh;
    int chosenDetector;
    float bestLargerDelta = SelectBestDetectorApprox(missCost,
                                                     falsePosCost,
                                                     timeCostPerError,
                                                     fracNegSampled,
                                                     *bestCascade,
                                                     winPastCascade,
                                                     &detectorQueue,
                                                     &chosenThresh,
                                                     &chosenDetector);
    if (bestLargerDelta < bestSmallerDelta && bestLargerDelta < 0) {
      bestCascade->AddStage(detectors_[chosenDetector], chosenThresh);
      foundBetterCascade = true;
      ROS_INFO_STREAM("Adding a stage with detector " << chosenDetector
                      << " of threshold " << chosenThresh
                      << " which changes cost by " << bestLargerDelta);
    } else if (bestSmallerDelta < 0) {
      bestCascade->EraseStage(stage2Remove);
      foundBetterCascade = true;
      ROS_INFO_STREAM("Removing a stage which changes cost by " <<
                      bestSmallerDelta);
    }
    
  }
  ioCascade->swap(bestCascade);
  return ComputeTotalCost(missCost, falsePosCost,
                          timeCostPerError, fracNegSampled,
                          **ioCascade);
  
}

double IntegralHogCascadeTrainer::ComputeTotalCost(
  float missCost,
  float falsePosCost,
  float timeCostPerError,
  float fracNegSampled,
  const IntegralHogCascade& cascade) {
  Mat windowsPassed = Mat_<uint8_t>::ones(1, scores_.cols);

  double curCost = timeCalculator_->GetIntegralHistTime() / timeCostPerError;
  double cpuTime = 0;

  Mat usefulWindows = (labels_ >= 0) & 0x1;
  Mat notUsefulWindows = (~usefulWindows) & 0x1;

  boost::unordered_set<cv::Size> cacheSizesUsed;

  // Step through the cascade and add up the computation costs for each stage
  for (vector<IntegralHogCascade::Stage>::const_iterator stageI =
         cascade.GetStageIterator();
       stageI != cascade.EndStages();
       ++stageI) {
    // Add the time to fill the cache
    if (cacheSizesUsed.find(stageI->second->blockSize()) ==
        cacheSizesUsed.end()) {
      cacheSizesUsed.insert(stageI->second->blockSize());
      cpuTime +=
        timeCalculator_->GetFillCacheTime(stageI->second->blockSize());
    }

    // Add the time to do the SVM evaluation
    double fracInputWindows =
      (sum(windowsPassed & usefulWindows)[0] +
       fracNegSampled * sum(windowsPassed & notUsefulWindows)[0]) /
      trueImageSum_;
    cpuTime += fracInputWindows * timeCalculator_->GetMaxWindows() *
      timeCalculator_->GetSVMEvalTime(stageI->second->descriptorSize());

    
    // Update which windows get past this stage
    windowsPassed = (windowsPassed & 
                     (scores_.row(detectorIdx_[stageI->second.get()]) >= 
                      stageI->first)) & 0x1;
  }

  // Calculate the computation time of running the high level algorithm
  double hogTime = timeCalculator_->GetTrueHogTime(
    ((float)timeCalculator_->GetMaxWindows()) *
    (sum(windowsPassed & usefulWindows)[0] +
     fracNegSampled * sum(windowsPassed & notUsefulWindows)[0]) /
    trueImageSum_);
  cpuTime += hogTime;

  curCost += cpuTime / timeCostPerError;

  // Now add the cost for misses
  int nMisses = sum((~windowsPassed & usefulWindows) & 0x1)[0];
  curCost += missCost * nMisses / nImages_;

  // Add the cost for the false positives
  int nFalsePos = sum((windowsPassed & notUsefulWindows) & 0x1)[0];
  curCost += falsePosCost * nFalsePos / nImages_;

  float nTruePos = sum((windowsPassed & usefulWindows) & 0x1)[0];

  ROS_INFO_STREAM("Precision: " << nTruePos / (nTruePos + nFalsePos)
                  << " Misses: " << nMisses
                  << " False Pos: " << nFalsePos
                  << " Accuracy: " << 1 - ((float)nFalsePos + nMisses) / scores_.cols
                  << " Hog Time: " << hogTime
                  << " Time: " << cpuTime + timeCalculator_->GetIntegralHistTime());
                                             
  return curCost;
}

void IntegralHogCascadeTrainer::InitializeApproxSubCost(
    float missCost,
    float falsePosCost,
    float timeCostPerError,
    float fracNegSampled,
    const IntegralHogCascade& cascade,
    boost::unordered_map<IntegralHogCascade::Stage, float>* removeCosts,
    cv::Mat* passedCascade) {
  ROS_ASSERT(passedCascade);
  ROS_ASSERT(removeCosts);

  Mat usefulWindows = (labels_ >= 0) & 0x1;
  Mat notUsefulWindows = (~usefulWindows) & 0x1;

  *passedCascade = Mat_<uint8_t>::ones(1, scores_.cols);

  // Get the counts of how many stages use each possible block
  // size. This is used so if a region is removed and it's the only
  // one with that block size, the cache doesn't need to be filled.
  unordered_map<Size, int> blockSizes;
  for (vector<IntegralHogCascade::Stage>::const_iterator stageI =
         cascade.GetStageIterator();
       stageI != cascade.EndStages();
       ++stageI) {
    const Size curBlockSize = stageI->second->blockSize();
    if (blockSizes.find(curBlockSize) == blockSizes.end()) {
      blockSizes[curBlockSize] = 1;
    } else {
      blockSizes[curBlockSize]++;
    }
  }

  for (vector<IntegralHogCascade::Stage>::const_iterator stageI =
         cascade.GetStageIterator();
       stageI != cascade.EndStages();
       ++stageI) {
    // Calculate the change in the D_a term, which is the direct
    // processing time of this stage.
    double fracInputWindows =
      ((float)sum(*passedCascade & usefulWindows)[0]+
       fracNegSampled * sum(*passedCascade & notUsefulWindows)[0]) /
      trueImageSum_;

    double deltaDa = -fracInputWindows * timeCalculator_->GetMaxWindows() * 
      timeCalculator_->GetSVMEvalTime(stageI->second->descriptorSize());
    
    // See if this means that we don't need to fill up a cache
    // Add the time to fill the cache
    if (blockSizes[stageI->second->blockSize()] == 1) {
      deltaDa -=
        timeCalculator_->GetFillCacheTime(stageI->second->blockSize());
    }

    // Add on the cost for computing more regions in later stages
    Mat passStage = (scores_.row(detectorIdx_[stageI->second.get()]) >=
                     stageI->first) & 0x1;
    Mat newlyFiltered = *passedCascade & ~passStage & 0x1;
    double newlyFilteredFrac =
      ((float)sum(newlyFiltered & usefulWindows)[0] +
       fracNegSampled * sum(newlyFiltered & notUsefulWindows)[0]) / 
      trueImageSum_;
    vector<IntegralHogCascade::Stage>::const_iterator stageJ(stageI);
    for (;
         stageJ != cascade.EndStages();
         ++stageJ) {
      deltaDa += newlyFilteredFrac * timeCalculator_->GetMaxWindows() *
        timeCalculator_->GetSVMEvalTime(stageJ->second->descriptorSize());
    }
    ROS_ASSERT(cascade.GetStageCount() == 0 || stageJ != stageI);

    // Calculate the extra misses from this stage
    Mat passWithoutThisStage = Mat_<uint8_t>::ones(1, scores_.cols);
    for (vector<IntegralHogCascade::Stage>::const_iterator
           stageJ = cascade.GetStageIterator();
         stageJ != cascade.EndStages();
         ++stageJ) {
      if (stageJ->second.get() != stageI->second.get()) {
        passWithoutThisStage &= (
          scores_.row(detectorIdx_[stageI->second.get()]) >=
          stageI->first) & 0x1;
      }
    }
    int deltaMisses = sum(~passStage & passWithoutThisStage &
                          usefulWindows & 0x1)[0];

    // Record the approximate cost change if this stage is removed.
    (*removeCosts)[*stageI] = -missCost * deltaMisses / nImages_ +
      deltaDa / timeCostPerError;

    // Update which regions have passed this stage 
    *passedCascade &= passStage;
  }
}

float IntegralHogCascadeTrainer::CalculateDeltaForRemovingStage(
  float falsePosCost,
  float timeCostPerError,
  float fracNegSampled,
  const IntegralHogCascade& cascade,
  const boost::unordered_map<IntegralHogCascade::Stage, float>& approxCosts,
  const IntegralHogCascade::Stage& stage) {

  Mat usefulWindows = (labels_ >= 0) & 0x1;
  Mat notUsefulWindows = (~usefulWindows) & 0x1;

  // Figure out which entries would pass with this stage and without it
  Mat passWithoutThisStage = Mat_<uint8_t>::ones(1, scores_.cols);
  Mat passCascade = Mat_<uint8_t>::ones(1, scores_.cols);
  for (vector<IntegralHogCascade::Stage>::const_iterator
         stageJ = cascade.GetStageIterator();
       stageJ != cascade.EndStages();
       ++stageJ) {
    Mat passStage = (scores_.row(detectorIdx_[stageJ->second.get()]) >=
        stageJ->first) & 0x1;
    if (stageJ->second.get() != stage.second.get()) {
      passWithoutThisStage &= passStage;
    }
    passCascade &= passStage;
  }

  Mat passStage = (scores_.row(detectorIdx_[stage.second.get()]) >=
                   stage.first) & 0x1;

  // Add the D_a and C_M approximate costs
  float deltaCost = approxCosts.find(stage)->second;

  // Add the D_r costs
  float nPassCascade = sum(passCascade & usefulWindows)[0] +
    fracNegSampled * sum(passCascade & notUsefulWindows)[0];
  float nPassWithoutStage = sum(passWithoutThisStage & usefulWindows)[0] +
    fracNegSampled * sum(passWithoutThisStage & notUsefulWindows)[0];
  float deltaCpu = timeCalculator_->GetTrueHogTime(
      ((float)timeCalculator_->GetMaxWindows()) * nPassWithoutStage /
      trueImageSum_) -
    timeCalculator_->GetTrueHogTime(
      ((float)timeCalculator_->GetMaxWindows()) * nPassCascade /
      trueImageSum_);
  deltaCost += deltaCpu /  timeCostPerError;

  // Add the C_F costs
  deltaCost += falsePosCost / nImages_ * 
    (sum(notUsefulWindows &
         passWithoutThisStage)[0] - 
     sum(notUsefulWindows & passCascade)[0]);

  ROS_INFO_STREAM("Delta for removing stage is " << deltaCost);
  return deltaCost;
}

float IntegralHogCascadeTrainer::SelectBestDetectorApprox(
  float missCost,
  float falsePosCost,
  float timeCostPerError,
  float fracNegSampled,
  const IntegralHogCascade& cascade,
  const cv::Mat& winPastApproxCascade,
  DetectorQueue* detectorQueue,
  float* thresh,
  int* chosenDetector) {
  ROS_ASSERT(thresh && chosenDetector && detectorQueue);
  *chosenDetector = -1;
  *thresh = 0;

  // Calculate the windows passsed the current cascade and keep track
  // of the block sizes in the cascade.
  Mat windowsPassed = Mat_<uint8_t>::ones(1, scores_.cols);
  for (vector<IntegralHogCascade::Stage>::const_iterator stageI =
         cascade.GetStageIterator();
       stageI != cascade.EndStages();
       ++stageI) {
    windowsPassed = (windowsPassed & 
                     (scores_.row(detectorIdx_[stageI->second.get()]) >= 
                      stageI->first)) & 0x1;
    ROS_INFO_STREAM("After stage " << detectorIdx_[stageI->second.get()] <<
                    ", " << sum(windowsPassed)[0] << " passed");
  }

  if (sum(windowsPassed)[0] < 2) {
    // Not enough windows pass for it to make sense to add a new filter
    return 0;
  }

  Mat usefulWindows = (labels_ >=0) & 0x1;
  Mat notUsefulWindows = (~usefulWindows) & 0x1;
  Mat usefulPassApproxCascade = usefulWindows & winPastApproxCascade;
  Mat falsePosWins = notUsefulWindows & windowsPassed & 0x1;

  float nPassedApproxCascade = 
    sum(winPastApproxCascade & usefulWindows)[0] +
    fracNegSampled * sum(winPastApproxCascade & notUsefulWindows)[0];

  float nWindowsPassed = 
    sum(windowsPassed & usefulWindows)[0] +
    fracNegSampled * sum(windowsPassed & notUsefulWindows)[0];
  double baseHogTime = timeCalculator_->GetTrueHogTime(
    nWindowsPassed * timeCalculator_->GetMaxWindows() / trueImageSum_);

  // If the detector queue is empty, we need to initialize it
  if (detectorQueue->empty()) {
    for (unsigned int i = 0u; i < detectors_.size(); ++i) {
      pair<float, float> bestThresh = GetBestApproxCost(
        i,
        missCost,
        falsePosCost,
        timeCostPerError,
        fracNegSampled,
        baseHogTime,
        nPassedApproxCascade,
        usefulPassApproxCascade,
        falsePosWins,
        windowsPassed,
        usefulWindows,
        notUsefulWindows);

      detectorQueue->push(DetectorStats(bestThresh.first, bestThresh.second,
                                        i));
    }
  }

  // Now, we pop of the top of the priority queue and update the cost
  // until we're still better than the second best choice
  while (true) {
    DetectorStats curDetector = detectorQueue->top();
    detectorQueue->pop();

    pair<float, float> bestThresh = GetBestApproxCost(
        curDetector.detectorIdx,
        missCost,
        falsePosCost,
        timeCostPerError,
        fracNegSampled,
        baseHogTime,
        nPassedApproxCascade,
        usefulPassApproxCascade,
        falsePosWins,
        windowsPassed,
        usefulWindows,
        notUsefulWindows);

    if (detectorQueue->empty() || 
        bestThresh.first <= 
        (detectorQueue->top().score + numeric_limits<float>::epsilon()*100)) {
      // Even after the update, this entry is the best, so use it
      *chosenDetector = curDetector.detectorIdx;
      *thresh = bestThresh.second;
      return bestThresh.first;
    }

    // Another entry is better, so update the score for this detector
    // and push it back into the queue.
    detectorQueue->push(DetectorStats(bestThresh.first, bestThresh.second,
                                      curDetector.detectorIdx));
  }

  return 0; // Should never get here.
}

std::pair<float, float> IntegralHogCascadeTrainer::GetBestApproxCost(
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
  const cv::Mat& notUsefulWindows) {

  float bestCost = numeric_limits<float>::infinity();
  float bestThresh = 0;

  const Mat_<float> curScores = scores_.row(detectorIdx);

  // Calculate the D_a term change by adding this detector
  float cpuDelta = nPassedApproxCascade / trueImageSum_ * 
    timeCalculator_->GetMaxWindows() *
    timeCalculator_->GetSVMEvalTime(
      detectors_[detectorIdx]->descriptorSize());
  // Add the time to fill the cache. This is an overestimate because
  // the approx cascade could have the cache already filled.
  cpuDelta += timeCalculator_->GetFillCacheTime(
    detectors_[detectorIdx]->blockSize());

  // Get the list of unique possible thresholds
  vector<float> validThresh(curScores.begin(), curScores.end());
  std::sort(validThresh.begin(), validThresh.end());
  vector<float>::const_iterator endUnique = 
    std::unique(validThresh.begin(), validThresh.end());

  // Check the thresholds to find the best one by reduction in
  // Visual Utility Risk.
  int j = 0;
  for(vector<float>::const_iterator threshI = validThresh.begin();
      threshI != endUnique; ++threshI) {
    // Speed this up by not checking every possible threshold. We'll
    // be close enough.
    if (++j % 4 != 0) {
      continue;
    }
    
    // The windows that would pass this stage
    Mat passStage = (curScores >= *threshI) & 0x1;

    // Calculate the number of new misses
    double nNewMisses = sum(usefulPassApproxCascade &
                            ~passStage)[0];
    
    // Calculate the change in the D_R term
    Mat passCur = passStage & windowsPassed;
    float nPassStage = sum(passCur & usefulWindows)[0] +
      fracNegSampled * sum(passCur & notUsefulWindows)[0];
    double deltaDRTime =        
      timeCalculator_->GetTrueHogTime(nPassStage *
                                      timeCalculator_->GetMaxWindows() /
                                      trueImageSum_) -
      baseHogTime;

    // Calculate the number of removed false positives
    double deltaFP = sum(falsePosWins & passStage)[0] -
      sum(falsePosWins)[0];
    
    double deltaCost = (cpuDelta + deltaDRTime) / timeCostPerError +
      missCost * nNewMisses / nImages_ +
      falsePosCost * deltaFP / nImages_;

    if (deltaCost < bestCost) {
      bestCost = deltaCost;
      bestThresh = *threshI;
    }
  }

  return std::pair<float, float>(bestCost, bestThresh);
}

struct EvalWrapperParams {
  EvalWrapperParams(const DeltaMinimizer* _obj, void* _params)
    : obj(_obj), params(_params) {}
  const DeltaMinimizer* obj;
  void* params;
};

double EvalCostWrapper(const gsl_vector* nFiltered, void* params) {
  EvalWrapperParams* p = reinterpret_cast<EvalWrapperParams*>(params);
  return p->obj->EvalCost(nFiltered, p->params);
}
void EvalDerivWrapper(const gsl_vector* nFiltered, void* params,
                      gsl_vector* derivs) {
  EvalWrapperParams* p = reinterpret_cast<EvalWrapperParams*>(params);
  p->obj->EvalDeriv(nFiltered, p->params, derivs);
}

void EvalWithDerivWrapper(const gsl_vector* nFiltered, void* params,
                          double* val, gsl_vector* derivs) {
  EvalWrapperParams* p = reinterpret_cast<EvalWrapperParams*>(params);
  p->obj->EvalWithDeriv(nFiltered, p->params, val, derivs);
}



DeltaMinimizer::DeltaMinimizer() : 
  bestCost_(0), bestThresh_(0), minimizer_(NULL) {
  minimizer_ = gsl_multimin_fdfminimizer_alloc(
    gsl_multimin_fdfminimizer_conjugate_fr, 1);
  minFunc_.f = &EvalCostWrapper;
  minFunc_.df = &EvalDerivWrapper;
  minFunc_.fdf = &EvalWithDerivWrapper;
  minFunc_.n = 1;
  minFunc_.params = NULL;

}

DeltaMinimizer::~DeltaMinimizer() {
  if (minimizer_ != NULL) {
    gsl_multimin_fdfminimizer_free(minimizer_);
  }
}

float DeltaMinimizer::MinimizeCost(
  float missCost, 
  float falsePosCost, 
  float timeCostPerError,
  float fracNegSampled,
  int nImages,
  float trueImageSum,
  double baseHogTime,
  const Mat_<float>& scores,
  const Mat& usefulPassApproxCascade,
  const Mat& falsePosWins,
  const Mat& windowsPassed,
  const Mat& usefulWindows,
  const Mat& notUsefulWindows,
  const IntegralCascadeTimeCalculator* timeCalculator,
  float cpuDelta) {
  cpuDelta_ = cpuDelta;

  const int maxTimeWindows = timeCalculator->GetMaxWindows();

  // First build the interpolators that we'll use. We can sample
  // pretty sparsely and interpolate because the false positive
  // windows are most of them so it is roughly linear with respect to
  // the number of windows thresholded.

  // First collect all the thresholds that cause a new miss based on
  // the current cascade
  vector<double> missThreshs;
  for (int i = 0; i < scores.cols; ++i) {
    if (windowsPassed.at<uint8_t>(i) && usefulWindows.at<uint8_t>(i)) {
      missThreshs.push_back(scores(i));
    }
  }
  std::sort(missThreshs.begin(), missThreshs.end());
  vector<double>::iterator endUnique = 
      std::unique(missThreshs.begin(), missThreshs.end());

  // Now, walk through all the possible thresholds and build up the
  // functions to interpolate.
  vector<double> nNewWinThresh;
  vector<double> deltaMissCost;
  vector<double> deltaFPCost;
  vector<double> deltaHogTimeCost;
  for(vector<double>::const_iterator threshI = missThreshs.begin();
      threshI != endUnique; ++threshI) {
    // The windows that would pass this stage
    Mat passStage = (scores >= *threshI) & 0x1;

    // Number of new windows thresholded.
    Mat newExamplesCut = ~passStage & windowsPassed;
    nNewWinThresh.push_back(
      sum(newExamplesCut & usefulWindows)[0] +
      fracNegSampled * sum(newExamplesCut & notUsefulWindows)[0]);

    // The cost of new misses
    double nNewMisses = sum(usefulPassApproxCascade &
                            ~passStage)[0];
    deltaMissCost.push_back(nNewMisses * missCost / nImages);

    // The cost of removed false positives
    double deltaFP = sum(falsePosWins & passStage)[0] -
        sum(falsePosWins)[0];
    deltaFPCost.push_back(deltaFP * falsePosCost / nImages);

    // The change in the D_R term
    Mat passCur = passStage & windowsPassed;
    double nPassStage = sum(passCur & usefulWindows)[0] +
      fracNegSampled * sum(passCur & notUsefulWindows)[0];
    double deltaDRTime =        
      timeCalculator->GetTrueHogTime(nPassStage *
                                     maxTimeWindows /
                                     trueImageSum) -
      baseHogTime;
    deltaHogTimeCost.push_back(deltaDRTime / timeCostPerError);
  }
  thresholdInterpolator_.reset(
    new SplineInterpolator(
      nNewWinThresh.begin(), nNewWinThresh.end(),
      missThreshs.begin(), endUnique));
  missCostInterpolator_.reset(new SplineInterpolator(nNewWinThresh,
                                                     deltaMissCost));
  falsePosInterpolator_.reset(new SplineInterpolator(nNewWinThresh,
                                                     deltaFPCost));
  hogTimeInterpolator_.reset(new SplineInterpolator(nNewWinThresh,
                                                    deltaHogTimeCost));

  // Now, do the minimization starting from the middle. We know that
  // the function has at most 2 humps in it in the valid range, so
  // we'll take advantage of that fact.
  float stepSize = scores.cols / 100.0;
  float middleCost;
  float middleThresh;
  RunMinimization(nNewWinThresh[0], nNewWinThresh.back(),
                  (nNewWinThresh[0] + nNewWinThresh.back()) / 2,
                  stepSize, &middleCost, &middleThresh);
  // Check to see if we hit the bounds. If so, we starting on top of a
  // hill, so try minimizing starting from the other side
  float sideCost;
  float sideThresh;
  if (middleThresh <= nNewWinThresh[0]) {
    RunMinimization(nNewWinThresh[0], nNewWinThresh.back(),
                    nNewWinThresh.back(),
                    -stepSize, &sideCost, &sideThresh);
  } else if (middleThresh >= nNewWinThresh.back()) {
    RunMinimization(nNewWinThresh[0], nNewWinThresh.back(),
                    nNewWinThresh[0],
                    stepSize, &sideCost, &sideThresh);
  }

  if (middleCost < sideCost) {
    bestCost_ = middleCost + cpuDelta_ / timeCostPerError;
    bestThresh_ = middleThresh;
  } else {
    bestCost_ = sideCost + cpuDelta_ / timeCostPerError;
    bestThresh_ = sideThresh;
  }

  return bestCost_;
}

void DeltaMinimizer::RunMinimization(double minWindowsThreshed,
                                     double maxWindowsThreshed,
                                     double startPoint,
                                     double firstStep,
                                     float* bestCost,
                                     float* bestThresh) {
  ROS_ASSERT(minWindowsThreshed < maxWindowsThreshed);
  ROS_ASSERT(startPoint >= minWindowsThreshed &&
             startPoint <= maxWindowsThreshed);

  // Load up the bounds for the function
  double bounds[2];
  bounds[0] = minWindowsThreshed;
  bounds[1] = maxWindowsThreshed;
  EvalWrapperParams params(this, &bounds);
  minFunc_.params = &params;

  // Do the minimization
  double curThresh = GSL_NAN;
  int status = GSL_CONTINUE;
  int iter = 0;
  gsl_vector* x = gsl_vector_alloc(1);
  gsl_vector_set(x, 0, startPoint);
  gsl_multimin_fdfminimizer_set(minimizer_, &minFunc_, x, firstStep, 0.1);
  do {
    status = gsl_multimin_fdfminimizer_iterate(minimizer_);
    curThresh = gsl_vector_get(minimizer_->x, 0);
    
    ROS_INFO_STREAM("Moved to threshold " << curThresh
                    << " With cost: " << minimizer_->f);

    if (status) {
      break;
    }

    status = gsl_multimin_test_gradient(minimizer_->gradient, 1e-5);

  } while (status == GSL_CONTINUE && ++iter < 50 && 
           curThresh >= minWindowsThreshed &&
           curThresh <= maxWindowsThreshed);

  gsl_vector_free(x);

  if (status != GSL_SUCCESS && status != GSL_CONTINUE &&
      status != GSL_ENOPROG) {
    ROS_ERROR_STREAM("Minimizer Error: " << gsl_strerror(status));
    *bestCost = numeric_limits<float>::infinity();
    return;
  }

  *bestCost = minimizer_->f;
  if (curThresh < minWindowsThreshed) {
    *bestThresh = (*thresholdInterpolator_)(minWindowsThreshed);
  } else if (curThresh > maxWindowsThreshed) {
    *bestThresh =(* thresholdInterpolator_)(maxWindowsThreshed);
  } else {
    *bestThresh = (*thresholdInterpolator_)(curThresh);
  }
  
  if (iter >= 50) {
    ROS_INFO_STREAM("Did not converge");
  } else {
    ROS_INFO_STREAM("Converged");
  }  
}

double DeltaMinimizer::EvalCost(const gsl_vector* nFiltered,
                                void* params) const {
  const double x = gsl_vector_get(nFiltered, 0);
  // gsl can't handle constrained minimiziation, so we flatten the
  // function outside of its bounds.
  double* bounds = reinterpret_cast<double*>(params);
  if (x < bounds[0]) {
    return missCostInterpolator_->minY() +
      falsePosInterpolator_->minY() +
      hogTimeInterpolator_->minY();
  } else if (x > bounds[1]) {
    return missCostInterpolator_->maxY() +
      falsePosInterpolator_->maxY() +
      hogTimeInterpolator_->maxY();
  }

  return (*missCostInterpolator_)(x) +
    (*falsePosInterpolator_)(x) +
    (*hogTimeInterpolator_)(x);
}
void DeltaMinimizer::EvalDeriv(const gsl_vector* nFiltered, void* params,
                               gsl_vector* derivs) const {
  ROS_ASSERT(derivs);
  const double x = gsl_vector_get(nFiltered, 0);

  // gsl can't handle constrained minimiziation, so we flatten the
  // function outside of its bounds.
  double* bounds = reinterpret_cast<double*>(params);
  if (x < bounds[0] || x > bounds[1]) {
    gsl_vector_set(derivs, 0, 0);
  } else {
    gsl_vector_set(derivs, 0, missCostInterpolator_->EvalDeriv(x) +
                 falsePosInterpolator_->EvalDeriv(x) +
                 hogTimeInterpolator_->EvalDeriv(x));
  }
}

void DeltaMinimizer::EvalWithDeriv(const gsl_vector* nFiltered, void* params,
                                   double* val, gsl_vector* derivs) const {
  ROS_ASSERT(val);
  ROS_ASSERT(derivs);

  *val = EvalCost(nFiltered, params);
  EvalDeriv(nFiltered, params, derivs);
}

} // namespace
