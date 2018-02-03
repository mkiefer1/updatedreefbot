#include "visual_utility/VisualUtilityEstimatorXMLParameters.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <ros/ros.h>

using boost::property_tree::ptree;
using namespace std;
using namespace boost;

namespace visual_utility {

typedef boost::shared_ptr<VisualUtilityEstimator> VisualUtilityEstimatorPtr;

VisualUtilityEstimator* ParseSingleEstimator(const ptree& node);

TransformEstimator* ParseTransformEstimator(const ptree& node) {
  // Get the class name
  string className;
  try {
    className = node.get<string>("class");
  } catch (property_tree::ptree_bad_path& e) {
    ROS_ERROR("Could not find the class name");
    return false;
  }

  if (className == "AffineTransformEstimator") {
    return new AffineTransformEstimator(node.get<int>("max_iterations"),
                                        node.get<double>("min_precision"),
                                        node.get<double>("image_scaling"));
  }
  ROS_ERROR_STREAM("Could not find the transform estimator: " << className);
  return NULL;
}

VisualUtilityEstimator* ParseLABMotionVUEstimator(const ptree& node) {
  TransformEstimator* transformEstimator =
    ParseTransformEstimator(node.get_child("transform_estimator"));
  if (transformEstimator == NULL) {
    return NULL;
  }
  return new LABMotionVUEstimator(
    *transformEstimator,
    node.get<double>("pareto_threshold"),
    node.get<double>("dist_decay"),
    node.get<int>("opening_size"));
}

VisualUtilityEstimator* ParseSpectralSaliency(const ptree& node) {
  return new SpectralSaliency();
}

VisualUtilityEstimator* ParseRelativeEntropyVUWrapper(const ptree& node) {
  return new RelativeEntropyVUWrapper(ParseSingleEstimator(
    node.get_child("base_estimator")));
}

VisualUtilityEstimator* ParseLaplacianVU(const ptree& node) {
  return new LaplacianVU(node.get<int>("k_size"));
}

VisualUtilityEstimator* ParseCenterSurroundHistogram(const ptree& node) {
  // Parse the list of scales
  vector<double> scales;
  BOOST_FOREACH(const ptree::value_type &v,
                node.get_child("scales")) {
    scales.push_back(v.second.get_value<double>());
  }

  return new CenterSurroundHistogram(scales,
                                     node.get<string>("dist_type"));
}

VisualUtilityEstimator* ParseObjectness(const ptree& node) {
  return new Objectness();
}

VisualUtilityEstimator* ParseROSBag(const ptree& node) {
  return new ROSBag(node.get<string>("filename"));
}

VisualUtilityEstimator* ParseHOGDetector(const ptree& node) {
  return new HOGDetector(node.get<string>("model_file"),
                         node.get<bool>("use_default_people_detector"));
}

VisualUtilityEstimator* ParseSingleEstimator(const ptree& node) {
  // Get the class name
  string className;
  try {
    className = node.get<string>("class");
  } catch (property_tree::ptree_bad_path& e) {
    ROS_ERROR("Could not find the class name");
    return NULL;
  }
  
  if (className == "LABMotionVUEstimator") {
    return ParseLABMotionVUEstimator(node);
  } else if (className == "SpectralSaliency") {
    return ParseSpectralSaliency(node);
  } else if (className == "RelativeEntropyVUWrapper") {
    return ParseRelativeEntropyVUWrapper(node);
  } else if (className == "LaplacianVU") {
    return ParseLaplacianVU(node);
  } else if (className == "CenterSurroundHistogram") {
    return ParseCenterSurroundHistogram(node);
  } else if (className == "Objectness") {
    return ParseObjectness(node);
  } else if (className == "ROSBag") {
    return ParseROSBag(node);
  } else if (className == "HOGDetector") {
    return ParseHOGDetector(node);
  }

  ROS_ERROR_STREAM("Invalid class name: " << className);

  return NULL;
}

bool ParseVUEstimatorsXMLFile(const string& filename,
                              VUEstimatorContainer* out) {
  ROS_ASSERT(out);
  ROS_INFO_STREAM("Parsing the XML file: " << filename);

  ptree tree;
  read_xml(filename, tree);
  BOOST_FOREACH(ptree::value_type &v,
                tree.get_child("estimators")) {
    VisualUtilityEstimator* estimator = ParseSingleEstimator(v.second);
    if (estimator == NULL) {
      ROS_ERROR_STREAM("There was an error parsing the file: " << filename);
      return false;
    }
    out->push_back(shared_ptr<VisualUtilityEstimator>(estimator));
  }

  return true;
}

} // namespace
