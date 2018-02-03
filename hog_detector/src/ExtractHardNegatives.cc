// A program that extracts any candidate windows from an image that
// first on a given integral hog detector. This allows you to extract
// hard negative examples.
//
// Usage: ExtractHardNegatives [options]
//
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
// Date: 2012

#include "ros/ros.h"
#include <gflags/gflags.h>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <boost/filesystem.hpp>

#include "hog_detector/integral_hog_detector_inl.h"

DEFINE_string(output_dir, "", "Directory to output the images to");
DEFINE_string(input, "", "Image to scan for positive hits in");
DEFINE_string(detector, "", "File specifying the detector to use");

DEFINE_int32(win_stride, 8, "Stride in the image to evaluate the detector");
DEFINE_double(thresh, 0.0, "Threshold for the detector to consider a hit");
DEFINE_int32(winH, 128, "Height of the cannoncial window");
DEFINE_int32(winW, 64, "Width of the cannoncial window");

using namespace hog_detector;
using namespace cv;
using namespace std;
namespace fs = boost::filesystem;

int main(int argc, char** argv) {
  // Parse the input
  google::ParseCommandLineFlags(&argc, &argv, true);

  ros::init(argc, argv, "integral_hog_trainer",
            ros::init_options::AnonymousName);

  // Open up the image
  Mat image = imread(FLAGS_input);
  if (image.empty()) {
    ROS_FATAL_STREAM("Could not open image " << FLAGS_input);
    exit(1);
  }

  // Open the detector
  IntegralHogDetector detector(FLAGS_detector, Size(FLAGS_win_stride,
                                                    FLAGS_win_stride));
  detector.SetThresh(FLAGS_thresh);

  // Put together the list of candidate regions
  vector<Rect> rois;
  for (int y = 0; y < image.rows; y += FLAGS_win_stride) {
    for (int x = 0; x < image.cols; x += FLAGS_win_stride) {
      rois.push_back(Rect(x, y, FLAGS_winW, FLAGS_winH));
    }
  }

  // Get all the regions where an image is detected incorrectly.
  vector<Rect> foundLocations;
  vector<double> scores;
  detector.DetectObjects(image, rois, &foundLocations, &scores);

  // Output all the iamges
  int curId = 0;
  for (vector<Rect>::const_iterator roiI = foundLocations.begin();
       roiI != foundLocations.end(); ++roiI) {
    fs::path inputPath(FLAGS_input);
    stringstream imgFn;
    imgFn << inputPath.filename() << "." << curId++ << ".jpg";
    imwrite((fs::path(FLAGS_output_dir) / imgFn.str()).string(), image(*roiI));
  }
}
