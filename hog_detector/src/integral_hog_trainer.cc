// A program that trains an integral HOG detector
//
// Usage: integral_hog_trainer [options] <posList> <negList>
//
// posList and negList are files that contain lists of images, one per
// line for the positive and negative examples respectively.
//
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
// Date: 2012

#include "ros/ros.h"
#include <gflags/gflags.h>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/random.hpp>
#include <iostream>
#include <fstream>

#include "hog_detector/integral_hog_detector_inl.h"

DEFINE_int32(winH, 128, "Height of the cannoncial window");
DEFINE_int32(winW, 64, "Width of the cannoncial window");
DEFINE_int32(blockH, 16, "Height of the HOG block");
DEFINE_int32(blockW, 16, "Width of the HOG block");
DEFINE_int32(blockStrideX, 8, "Stride in the x of the HOG block");
DEFINE_int32(blockStrideY, 8, "Stride in the y of the HOG block");
DEFINE_int32(cellH, 8, "Height of the HOG cell");
DEFINE_int32(cellW, 8, "Width of the HOG cell");
DEFINE_int32(nbins, 9, "Number of orientation bins in the histogram");
DEFINE_int32(subWinX, 0, "X offset of the subwindow for this detector");
DEFINE_int32(subWinY, 0, "Y offset of the subwindow for this detector");
DEFINE_int32(subWinW, 64, "Width of the subwindow for this detector");
DEFINE_int32(subWinH, 128, "Height of the subwindow for this detector");

DEFINE_bool(sampleNegs, false, "If true, the negative examples will be sampled randomly first, and then we'll do hard negative learning");
DEFINE_int32(nSamples, 10, "Number of negative samples to take per image");
DEFINE_int32(seed, 165498, "Random number generator seed");

DEFINE_string(output, "integral_hog_detector.xml",
              "Output filename for the trained detector");

DEFINE_string(posTest, "", "File with the list of positive test examples");
DEFINE_string(negTest, "", "File with the list of negative test examples");

DEFINE_double(checkFrac, 0.2,
  "Fraction of examples to use when checking the training accuracy");

using namespace hog_detector;
using namespace cv;
using namespace std;
using namespace boost;

float GetScoreForImage(const string& imageFile,
                       const IntegralHogDetector& detector,
                       mt19937& randomGen,
                       bool randomRoi = false) {
  vector<Rect> foundLocations;
  vector<double> scores;

  Mat image = imread(imageFile);

  Rect roi;
  if (randomRoi) {
    uniform_int<> xSelector(0, image.cols - FLAGS_winW);
    uniform_int<> ySelector(0, image.rows - FLAGS_winH);

    roi = Rect(xSelector(randomGen),
               ySelector(randomGen),
               FLAGS_winW,
               FLAGS_winH);    
  } else {
    roi = Rect((image.cols - FLAGS_winW) / 2,
               (image.rows - FLAGS_winH) / 2,
               FLAGS_winW,
               FLAGS_winH);
  }
  
  detector.DetectObjects(image, 
                         vector<Rect>(1, roi),
                         &foundLocations,
                         &scores);
  return scores[0];
}

void CheckAccuracy(const vector<string>& posList,
                   const vector<string>& negList,
                   const IntegralHogDetector& detector,
                   mt19937& randomGen,
                   float checkFrac) {
  ROS_INFO_STREAM("Checking the accuracy of the detector");

  int nNegSamples = FLAGS_sampleNegs ?  FLAGS_nSamples : 1;

  unsigned int nImages = (posList.size()+negList.size()) * 
    checkFrac;
  int nCorrect = 0;
  int imageCount = 0;
  
  for (unsigned int i = 0u; i < nImages;) {
    if (i/2 < posList.size()) {
      float scorePos = GetScoreForImage(posList[i/2], detector, randomGen);
      if (scorePos > 0) {
        nCorrect++;
      }
      ++imageCount;
      ++i;
    }

    if (i/2 < negList.size()) {
      for (int j = 0; j < nNegSamples; ++j) {
        float scoreNeg = GetScoreForImage(negList[i/2], detector, randomGen,
                                          FLAGS_sampleNegs);
        if (scoreNeg < 0) {
          nCorrect++;
        }
        ++imageCount;
      }
      ++i;
    }
  }

  ROS_INFO_STREAM("Accuracy: "
                  << nCorrect << "/" << imageCount
                  << " (" << (100.0*nCorrect)/imageCount << "%)");
}

void ExtractImageList(const string& filename,
                      vector<string>* imageList) {
  ROS_ASSERT(imageList);

  char buf[512];

  ROS_INFO_STREAM("Opening " << filename << " to get image list");
  ifstream inStream(filename.c_str(), ios_base::in);
  while(!inStream.eof() && inStream.good()) {
    inStream.getline(buf, 512);
    if (!inStream.fail() && !inStream.bad()) {
      imageList->push_back(string(buf));
    }
  }

  ROS_INFO_STREAM("Found " << imageList->size() << " image files");
  
}

// Adds the center of the images to the detector for training
void AddTrainingEntriesFromCenter(IntegralHogDetector* detector,
                                  const vector<string>& imageFiles,
                                  float label) {
  for (vector<string>::const_iterator imageFile = imageFiles.begin();
       imageFile != imageFiles.end(); ++imageFile) {
    ROS_DEBUG_STREAM("Extracting HOG descriptors from " << *imageFile);

    // Start by opening the image
    Mat image = imread(*imageFile);

    // Find the winSize_ window in the center of the image
    cv::Rect win((image.cols - FLAGS_winW) / 2,
                 (image.rows - FLAGS_winH) / 2,
                 FLAGS_winW,
                 FLAGS_winH);

    detector->AddRegionsForTraining(image, vector<Rect>(1, win),
                                    vector<float>(1, label));

  }
}

void AddRandomWindowsFromImage(IntegralHogDetector* detector,
                               const vector<string>& imageFiles,
                               float label,
                               int nSamples,
                               mt19937& randomGen) {

  for (vector<string>::const_iterator imageFile = imageFiles.begin();
       imageFile != imageFiles.end(); ++imageFile) {
    ROS_DEBUG_STREAM("Extracting random frames from " << *imageFile);

    // Start by opening the image
    Mat image = imread(*imageFile);

    uniform_int<> xSelector(0, image.cols - FLAGS_winW);
    uniform_int<> ySelector(0, image.rows - FLAGS_winH);

    // Make a list of rois to add
    vector<Rect> rois;
    for (int i = 0; i < nSamples; ++i) {
      rois.push_back(Rect(xSelector(randomGen),
                          ySelector(randomGen),
                          FLAGS_winW,
                          FLAGS_winH));
    }

    detector->AddRegionsForTraining(image, rois, vector<float>(rois.size(),
                                                               label));
  }
}

void AddHardNegatives(IntegralHogDetector* detector,
                      const vector<string>& imageFiles,
                      unsigned int maxSamples) {

  detector->SetThresh(0);

  for (vector<string>::const_iterator imageFile = imageFiles.begin();
       imageFile != imageFiles.end(); ++imageFile) {
    ROS_DEBUG_STREAM("Extracting Hard Negatives From " << *imageFile);

    // Start by opening the image
    Mat image = imread(*imageFile);

    // Generate the set of candidate rois.
    vector<Rect> rois;
    for (int y = 0; y < image.rows - FLAGS_winH; y += FLAGS_blockStrideY) {
      for (int x = 0; x < image.cols - FLAGS_winW; x += FLAGS_blockStrideX) {
        rois.push_back(Rect(x, y, FLAGS_winW, FLAGS_winH));
      }
    }

    // Look for false positives
    vector<Rect> foundLocations;
    vector<double> scores;
    detector->DetectObjects(image, rois, &foundLocations, &scores);

    if (foundLocations.size() > maxSamples) {
      foundLocations.resize(maxSamples);
    }

    // Add the regions found to the training examples
    detector->AddRegionsForTraining(image, foundLocations,
                                    vector<float>(foundLocations.size(), -1),
                                    true); // Add to the front of the training
  }

  detector->SetThresh(-std::numeric_limits<float>::infinity());
}
                               

int main(int argc, char** argv) {
  // Parse the input
  google::ParseCommandLineFlags(&argc, &argv, true);

  ros::init(argc, argv, "integral_hog_trainer",
            ros::init_options::AnonymousName);

  if (argc != 3) {
    ROS_FATAL("Usage: integral_hog_trainer [options] <posList> <negList>");
    return 1;
  }

  // Create the random number generator
  mt19937 randomGen;
  randomGen.seed(FLAGS_seed);

  // Build up the list of entries to train on
  vector<string> posList;
  vector<string> negList;
  ExtractImageList(argv[1], &posList);
  ExtractImageList(argv[2], &negList);

  IntegralHogDetector detector(Size(FLAGS_winW, FLAGS_winH),
                               Size(FLAGS_blockW, FLAGS_blockH),
                               Size(FLAGS_blockStrideX, FLAGS_blockStrideY),
                               Size(FLAGS_cellW, FLAGS_cellH),
                               FLAGS_nbins,
                               -std::numeric_limits<float>::infinity(),
                               Rect(FLAGS_subWinX, FLAGS_subWinY,
                                    FLAGS_subWinW, FLAGS_subWinH));

  // Add the negative entries for training
  if (FLAGS_sampleNegs) {
    AddRandomWindowsFromImage(&detector, negList, 1.0, FLAGS_nSamples,
                              randomGen);
  } else {
    AddTrainingEntriesFromCenter(&detector, negList, 1.0);
  }

  // Add the positive entries for training
  AddTrainingEntriesFromCenter(&detector, posList, -1.0);
                               
  detector.DoTraining();

  if (FLAGS_sampleNegs) {
    ROS_INFO("Doing hard negative training");

    AddHardNegatives(&detector, negList, FLAGS_nSamples);

    detector.DoTraining();
  }

  ROS_INFO("Checking training accuracy");
  CheckAccuracy(posList, negList, detector, randomGen, FLAGS_checkFrac);

  if (!FLAGS_posTest.empty() && !FLAGS_negTest.empty()) {
    ROS_INFO("Checking positive test accuracy");
    vector<string> posTest;
    ExtractImageList(FLAGS_posTest, &posTest);

    CheckAccuracy(posTest, vector<string>(), detector, randomGen, 1.0);

    ROS_INFO("Checking negative test accuracy");
    vector<string> negTest;
    ExtractImageList(FLAGS_negTest, &negTest);

    CheckAccuracy(vector<string>(), negTest, detector, randomGen, 1.0);
  }


  ROS_INFO_STREAM("Saving the trained classifier to " << FLAGS_output);
  detector.save(FLAGS_output);
}
