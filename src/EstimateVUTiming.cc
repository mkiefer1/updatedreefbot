// Program that estimates the visual utility timing for one estimator
// and spits out a file where each line is of the form:
//
// <nWindows>,<seconds>
//
// Usage: EstimateVUTiming [options] <outputFile> <sampleImage0> <sampleImage1> ... <sampleImageN>

#include "visual_utility/TimingEstimator.h"
#include "visual_utility/VisualUtilityEstimator.h"
#include "visual_utility/VisualUtilityROSParams.h"
#include <ros/ros.h>
#include <fstream>
#include <boost/scoped_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>
#include <math.h>

using namespace std;
using namespace visual_utility;
using namespace boost;
using namespace cv;

int main(int argc, char **argv) {
  ros::init(argc, argv, "EstimateVUTiming",
            ros::init_options::AnonymousName);

  string outputFile(argv[1]);
  
  ros::NodeHandle handle("~");

  // Set the stride to enable caching in the HOG detector
  double winStride = 8;
  handle.setParam("win_stride", (int)winStride);
  bool useCache;
  handle.param<bool>("use_cache", useCache, true);
  handle.setParam("hog_do_cache", useCache);

  // Create the visual utility estimator we want to use
  scoped_ptr<TransformEstimator> transformEstimator(
    CreateTransformEstimator(handle));
  scoped_ptr<VisualUtilityEstimator> vuEstimator(
    CreateVisualUtilityEstimator(handle, *transformEstimator));

  // Grab the list of images
  vector<string> imageFiles;
  for (int i = 2; i < argc; ++i) {
    imageFiles.push_back(string(argv[i]));
  }

  // Create a list of rectangles to search for
  Mat image = imread(imageFiles[0]);
  // TODO(mdesnoyer): Make these parameters so that they aren't hard
  // coded for people detection.
  double curHeight = 128;
  double curWidth = 64;
  double scaleStride = 1.10;
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

  // Now build the timing estimator and figure out the timing
  int nSamples;
  handle.param<int>("samples", nSamples, 100);
  TimingEstimator timingEstimator(nSamples);
  ROS_INFO_STREAM("Starting to time the visual utility estimator");
  timingEstimator.LearnVUEstimatorTiming(vuEstimator.get(),
                                         imageFiles,
                                         rois);

  // Print to the output file
  ROS_INFO_STREAM("Outputing results to " << outputFile);
  fstream outStream(outputFile.c_str(),
                    ios_base::out | ios_base::trunc);
  timingEstimator.OutputToStream(outStream);
  outStream.close();

  return 0;
}
