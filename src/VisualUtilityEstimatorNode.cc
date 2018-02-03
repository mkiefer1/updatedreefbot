// A node that wraps the nodelet so that we can do some debugging or
// run directly from the command line.

#include "visual_utility/VisualUtilityEstimatorNodelet.h"

#include <string>

using namespace std;
using namespace ros;
using namespace visual_utility;

int main(int argc, char** argv) {
  ros::init(argc, argv, "VisualUtilityEstimatorNode");

  VisualUtilityEstimatorNodelet nodelet;

  nodelet.onInit();

  ros::spin();
}
