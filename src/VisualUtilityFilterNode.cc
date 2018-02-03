// A node that wraps the nodelet so that we can do some debugging

#include "visual_utility/VisualUtilityFilterNodelet.h"

#include <string>

using namespace std;
using namespace ros;
using namespace visual_utility;

int main(int argc, char** argv) {
  ros::init(argc, argv, "VisualUtilityFilterNode");

  VisualUtilityFilterNodelet nodelet;

  nodelet.onInit();

  ros::spin();
}
