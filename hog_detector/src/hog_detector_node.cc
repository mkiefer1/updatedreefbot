#include "hog_detector/hog_detector_nodelet.h"

using namespace std;

int main(int argc, char** argv) {
  ros::init(argc, argv, "HogDetector");

  hog_detector::HogDetectorNodelet nodelet;
  
  nodelet.onInit();

  ros::spin();
}
