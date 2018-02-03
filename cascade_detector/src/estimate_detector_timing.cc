// A script that estimates the runtime of the detector. Outputs a file
// where line n is the average time to process an image with the
// detector truncated at that number of cascade levels.
//
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
// Date: May 2012
//
// Usage: estimate_detector_timing <model_file> <output_file> <image0> ...

#include <ros/ros.h>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cascade_detector/detailed_cascade_detector.h"

using namespace cv;
using namespace std;
using namespace cascade_detector;

int main(int argc, char **argv) {

  ros::Time::init();

  // Parse the arguments
  ROS_ASSERT(argc > 3);
  string modelFile(argv[1]);
  string outputFile(argv[2]);
  vector<string> imageFiles;
  for (int i = 3; i < argc; ++i) {
    imageFiles.push_back(string(argv[i]));
  }

  // Create a list of rectangles to search for
  // Create a list of rectangles to search for
  Mat image = imread(imageFiles[0]);
  // TODO(mdesnoyer): Make these parameters so that they aren't hard
  // coded for people detection.
  double curHeight = 128;
  double curWidth = 64;
  double scaleStride = 1.10;
  double winStride = 8;
  vector<Rect> rois;
  while (curWidth < image.rows && curHeight < image.cols) {
    for (double x = 0; x < image.cols - curWidth; x += winStride) {
      for (double y = 0; y < image.rows - curHeight; y += winStride) {
        rois.push_back(Rect(round(x),
                            round(y),
                            round(curWidth),
                            round(curHeight)));
      }
    }
    curWidth *= scaleStride;
    curHeight *= scaleStride;
    winStride *= scaleStride;
  }

  DetailedCascadeDetector detector;
  detector.Init(modelFile);

  // We use this to accumulate the time to compute an image assuming
  // that it is cut off at a given stage.
  vector<double> samples;//(detector.GetNumCascadeStages(), 0.0);
  for (unsigned int i = 0u; i <= detector.GetNumCascadeStages(); ++i) {
    samples.push_back(0.0);
  }
  vector<int> scores;

  for (vector<string>::const_iterator imagePtr = imageFiles.begin();
       imagePtr != imageFiles.end(); ++imagePtr) {
    ROS_INFO_STREAM("Opening " << *imagePtr);
    Mat cvImage = imread(*imagePtr);

    // Start by running the detector and getting the level in the
    // cascade each window was thrown out on.
    vector<Rect> foundLocations;
    detector.DetectObjects(cvImage, rois, &foundLocations, &scores, NULL,
                           NULL);

    // Loop through all the possible stages
    int nBoxesCut = 0;
    double cutBoxesTiming = 0.0;
    double integralTime = -1;
    for (int stage = 0; stage <= detector.GetNumCascadeStages(); ++stage) {

      // Calculate the timing for all those boxes that get dropped at
      // this stage.
      vector<Rect> curRois;
      for (unsigned int i = 0u; i < rois.size(); ++i) {
        if (scores[i] == stage) {
          curRois.push_back(rois[i]);
        }
      }
      double curProcessingTime = 0;
      double tmpIntTime;
      vector<Rect> garb;
      detector.DetectObjects(cvImage, curRois, &garb, NULL,
                             &curProcessingTime,
                             &tmpIntTime);

      if (integralTime < 0) {
        integralTime = tmpIntTime;
      }

      double boxTiming = curProcessingTime - tmpIntTime;
      nBoxesCut += curRois.size();
      // Add the time to compute the integral and the time spent in
      // cutting boxes in previous stages.
      samples[stage] += integralTime + cutBoxesTiming;

      if (curRois.size() > 0) {
        // Now add the time to cut the boxes in this stage and the
        // computation of this stage for all boxes that make it past
        // this stage.
        samples[stage] +=  boxTiming + 
          boxTiming / curRois.size() * (rois.size() - nBoxesCut);
      }
      cutBoxesTiming += boxTiming;
    }
  }

  // Normalize the timing
  for (unsigned int i = 0u; i < samples.size(); ++i) {
    samples[i] /= imageFiles.size();
  }

  // Output the results
  ROS_INFO_STREAM("Outputing the results to " << outputFile);
  fstream outStream(outputFile.c_str(),
                    ios_base::out | ios_base::trunc);
  for (unsigned int i = 0u; i < samples.size(); ++i) {
    outStream << i << ',' << samples[i] << setprecision(17) << std::endl;
  }
  outStream.close();

  return 0;
  
}
