// Routine that trains an integral cascade using a set of images where
// the window of interest is in the center.
//
// Usage: TrainCascadeUsingImages [options] <posList> <negList> <detectorList>
//
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
// Date: Oct 2012


#include "ros/ros.h"
#include <gflags/gflags.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <boost/scoped_ptr.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "hog_detector/integral_hog_cascade_trainer.h"
#include "hog_detector/hog_detector_internal.h"

DEFINE_int32(winH, 128, "Height of the cannoncial window");
DEFINE_int32(winW, 64, "Width of the cannoncial window");
DEFINE_double(checkFrac, 0.2,
  "Fraction of examples to use when checking the training accuracy");
DEFINE_double(time_budget, 1.0,
              "Budget in seconds to prcoess an average image");
DEFINE_double(miss_cost, 0.5, "The cost of a miss");
DEFINE_double(false_pos_cost, 0.5, "The cost of a false positive "
              "hitting the high level HOG detector");
DEFINE_double(time_cost_per_error, 1.0,
              "The cost of 1 second of average processing per error");
DEFINE_double(hog_thresh, 0.0,
              "Threshold for whether the full HOG detector passes an example");

// Input flags
DEFINE_string(integral_hist_time, "",
              "File containing a single line specifying the time it takes "
              "to copute an integral histogram");
DEFINE_string(fill_block_cache_time, "", 
              "File containing lines <blockW>,<blockH>,<time>");
DEFINE_string(svm_eval_time, "", "File containing lines "
              "<descriptorSize>,<subWinW>,<subWinH>,<time/window>");
DEFINE_string(true_hog_time, "", "File containing lines "
              "<nWindows>,<time>");

// The output flags
DEFINE_string(output, "integral_hog_cascade.xml",
              "Output filename for the trained cascade");

using namespace hog_detector;
using namespace cv;
using namespace std;
using namespace boost;

void ExtractFileList(const string& filename,
                      vector<string>* fileList) {
  ROS_ASSERT(fileList);

  char buf[512];

  ROS_INFO_STREAM("Opening " << filename << " to get file list");
  ifstream inStream(filename.c_str(), ios_base::in);
  while(!inStream.eof() && inStream.good()) {
    inStream.getline(buf, 512);
    if (!inStream.fail() && !inStream.bad()) {
      fileList->push_back(string(buf));
    }
  }

  ROS_INFO_STREAM("Found " << fileList->size() << " files");
  
}

float GetScoreForImage(const string& imageFile,
                       const IntegralHogCascade& detector) {

  Mat image = imread(imageFile);

  Mat_<float> histSum;
  scoped_ptr<cv_utils::IntegralHistogram<float> > hist(
    detector.ComputeGradientIntegralHistograms(image, &histSum));

  return detector.ComputeScore(*hist, histSum,
                               Rect((image.cols - FLAGS_winW) / 2,
                                    (image.rows - FLAGS_winH) / 2,
                                    FLAGS_winW,
                                    FLAGS_winH));
}

void CheckTrainingAccuracy(const vector<string>& imageList,
                           const vector<float>& labels,
                           const IntegralHogCascade& detector) {
  ROS_INFO_STREAM("Checking the training accuracy of the detector");

  int nImages = imageList.size() * FLAGS_checkFrac;
  int nCorrect = 0;
  
  for (int i = 0; i < nImages;) {
    float scorePos = GetScoreForImage(imageList[i], detector);
    float scoreNeg = GetScoreForImage(imageList[imageList.size()-i-1],
                                      detector);
    if ((labels[i] > 0) == (scorePos == detector.GetStageCount())) {
      nCorrect++;
    }
    if ((labels[imageList.size()-1-i] > 0) == 
        (scoreNeg == detector.GetStageCount())) {
      nCorrect++;
    }
    i += 2;
  }

  ROS_INFO_STREAM("Training accuracy: "
                  << nCorrect << "/" << nImages
                  << " (" << (100.0*nCorrect)/nImages << "%)");
}

int main(int argc, char** argv) {
  // Parse the input
  google::ParseCommandLineFlags(&argc, &argv, true);

  ros::init(argc, argv, "integral_hog_trainer",
            ros::init_options::AnonymousName);

  if (argc != 4) {
    ROS_FATAL("Usage: integral_hog_trainer [options] <posList> <negList> <detector_list>");
    return 1;
  }

  // Build up the list of entries to train on
  vector<string> posList;
  vector<string> negList;
  ExtractFileList(argv[1], &posList);
  ExtractFileList(argv[2], &negList);
  vector<float> trainLabels(posList.size(), 1.0);
  trainLabels.insert(trainLabels.end(), negList.size(), -1.0);
  vector<string> trainList = posList;
  trainList.insert(trainList.end(), negList.begin(), negList.end());

  // Build the trainer
  IntegralHogCascadeTrainer trainer(
    new IntegralCascadeTimeCalculator(FLAGS_integral_hist_time,
                                      FLAGS_fill_block_cache_time,
                                      FLAGS_svm_eval_time,
                                      FLAGS_true_hog_time),
    Size(FLAGS_winW, FLAGS_winH));

  ROS_INFO("Adding all the candidate detectors");
  vector<string> detectorFiles;
  ExtractFileList(argv[3], &detectorFiles);
  for (vector<string>::const_iterator i = detectorFiles.begin();
       i != detectorFiles.end();
       ++i) {
    trainer.AddCandidateDetector(IntegralHogDetector::Ptr(
      new IntegralHogDetector(*i, Size(8,8))));
  }

  // Now do the training
  //trainer.Train(trainList,
  //              trainLabels,
  //             FLAGS_time_budget,
  //             FLAGS_miss_cost,
  //             FLAGS_false_pos_cost);
  trainer.TrainSupSub(trainList,
                      trainLabels,
                      FLAGS_miss_cost,
                      FLAGS_false_pos_cost,
                      FLAGS_time_cost_per_error,
                      1.0);

  CheckTrainingAccuracy(trainList, trainLabels, *trainer.cascade());

  ROS_INFO_STREAM("Saving the cascade to: " << FLAGS_output);
  trainer.cascade()->save(FLAGS_output);
}
