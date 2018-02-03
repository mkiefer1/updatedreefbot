// A set of structures to read xml definitions of visual utility
// estimators and create the corresponding objects.
//
// The xml file is in the format:
// <estimators>
//   <estimator>
//     <class>LABMotionVUEstimator</class>
//     ... parameters ...
//   </extimator>
//   ... more estimators ...
// </estimators>
//
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
// Date: April 2012

#ifndef __VISUAL_UTILITY_ESTIMATOR_XML_PARAMETERS_H__
#define __VISUAL_UTILITY_ESTIMATOR_XML_PARAMETERS_H__

#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include "VisualUtilityEstimator.h"

namespace visual_utility {

typedef std::vector<boost::shared_ptr<VisualUtilityEstimator> > VUEstimatorContainer;

// Function that parses an xml file and return a set of
// VisualUtilityEstimators.
//
// Inputs:
// filename - Filename for the XML file to parse
// out - Container to return the set of VisualUtilityEstimators
//
// Return value:
// true if the parsing was sucessfull
bool ParseVUEstimatorsXMLFile(const std::string& filename,
                              VUEstimatorContainer* out);

} // namespace

#endif // __VISUAL_UTILITY_ESTIMATOR_XML_PARAMETERS_H__
