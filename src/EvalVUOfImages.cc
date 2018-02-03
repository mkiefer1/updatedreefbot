// Program that evaluates the visual utility of a set of images. By
// default evaluates the window at the center of the image.
//
// Input file:
// <imageFilename>
//
// Output file:
// <imageFilename>,<score>
//
// Usage:
// EvalVUOfImages [options]

#include "visual_utility/VisualUtilityEstimator.h"
#include "visual_utility/VisualUtilityROSParams.h"
#include <ros/ros.h>
#include <fstream>
#include <iostream>
#include <boost/scoped_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>
#include <gflags/gflags.h>

DEFINE_int32(winH, 128, "Height of the cannoncial window");
DEFINE_int32(winW, 64, "Width of the cannoncial window");

DEFINE_string(input, "", "Input file to use. STDIN if not specified");
DEFINE_string(output, "", "Output file to write to. STDOUT if not specified");

DEFINE_double(thresh, 0.0, "Threshold for accepting an example. Used for stats");
DEFINE_bool(print_stats, false, "Optionally print the statistics at the end");

using namespace std;
using namespace visual_utility;
using namespace boost;
using namespace cv;

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  ros::init(argc, argv, "EvalVUOfImages",
            ros::init_options::AnonymousName);

  if (argc != 1) {
    ROS_FATAL("Usage: EvalVUOfImages [options]");
    return 1;
  }

  // Create the visual utility estimator we want to use
  ros::NodeHandle handle("~");
  scoped_ptr<TransformEstimator> transformEstimator(
    CreateTransformEstimator(handle));
  scoped_ptr<VisualUtilityEstimator> vuEstimator(
    CreateVisualUtilityEstimator(handle, *transformEstimator));

  // Open the file for reading the image names
  istream* inputStream = &std::cin;
  ifstream fileInputStream;
  if (!FLAGS_input.empty()) {
    fileInputStream.open(FLAGS_input.c_str(), ios_base::in);
    if (!fileInputStream.good()) {
      ROS_ERROR_STREAM("Could not open " << FLAGS_input);
      return 1;
    }
    inputStream = &fileInputStream;
  }

  // Open the file for output
  ostream* outputStream = &std::cout;
  ofstream fileOutputStream;
  if (!FLAGS_output.empty()) {
    fileOutputStream.open(FLAGS_output.c_str(),
                         ios_base::out | ios_base::trunc);
    if (!fileOutputStream.good()) {
      ROS_ERROR_STREAM("Could not open " << FLAGS_output);
      return 1;
    }
    outputStream = &fileOutputStream;
  }

  // Process one file at a time
  int nPositive = 0;
  int nExamples = 0;
  while(inputStream->good() && !inputStream->eof()) {
    string imageFile;
    (*inputStream) >> imageFile;

    if (imageFile.empty()) {
      break;
    }

    Mat image = imread(imageFile);

    if (image.empty()) {
      ROS_ERROR_STREAM("Could not open image: " << imageFile);
      continue;
    }

    vector<Rect> roi(1, Rect((image.cols - FLAGS_winW) / 2,
                             (image.rows - FLAGS_winH) / 2,
                             FLAGS_winW,
                             FLAGS_winH));
    vector<pair<double, Rect> > answer;
    vuEstimator->CalculateVisualUtility(image, roi, 0.0, &answer);

    (*outputStream) << imageFile << "," << answer[0].first << endl;

    nExamples++;
    if(answer[0].first > FLAGS_thresh) {
      nPositive++;
    }
  }

  fileOutputStream.close();
  fileInputStream.close();

  if (FLAGS_print_stats) {
    cout << "Accuracy: " << ((double)nPositive) / nExamples 
         << " (" << nPositive << "/" << nExamples << ")" << std::endl;
  }
  

  return 0;
}
