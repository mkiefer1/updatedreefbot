// Program that trains a visual utility cascade.  It takes as input a
// set of estimators in XML format and ouputs the definition of the
// resulting cascade in another XML file.
//
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
// Date: April, 2012
//
// Usage: CascadeTrainer [ros param options] <inputXML> <outputXML>

#include <ros/ros.h>
#include <string>
#include "visual_utility/VisualUtilityEstimatorXMLParameters.h"
#include "visual_utility/VisualUtilityEstimator.h"

using namespace std;
using namespace ros;
using namespace visual_utility;

int main(int argc, char**argv) {
  ros::init(argc, argv, CascadeTrainer);

  ROS_ASSERT(argc >= 3);

  // First load up the visual utility estimators
  VUEstimatorContainer estimators;
  ROS_ASSERT(ParseVUEstimatorsXMLFile(argv[1], &estimators));

  // Now calculate the timing estimates for each estimator

  // Get the query boxes

  // Evaluate each feature for each query box

  // Learn the cascade

  // Output the cascade to xml format
}

