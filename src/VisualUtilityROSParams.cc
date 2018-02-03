// Copyright 2012 Carnegie Mellon University
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
//
// VisualUtilityROSParams.h
//
// Utilities to build the objects of the visual utility pipeline from
// ROS parameters.

#include "visual_utility/VisualUtilityROSParams.h"

#include <string>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>

using namespace std;
using namespace cv;

namespace visual_utility {

// Function that parses a comma separated string into a vector of some
// datatype
template <typename T>
vector<T> ParseDataList(const string& str) {
  vector<T> retval;

  typedef boost::tokenizer<boost::char_separator<char> > Tokenizer;
  Tokenizer tokenizer(str, boost::char_separator<char>(", \t\n"));
  
  for (Tokenizer::iterator i = tokenizer.begin(); i != tokenizer.end(); ++i) {
    try {
      retval.push_back(boost::lexical_cast<T>(*i));
    } catch (boost::bad_lexical_cast& e) {
      ROS_ERROR_STREAM("There was an issue parsing the list: "
                       << str);
    }
  }
  return retval;
}


TransformEstimator* CreateTransformEstimator(ros::NodeHandle handle) {
  int maxIterations;
  double minPrecision;
  double imageScaling;
  handle.param<int>("affine_max_iterations", maxIterations, 100);
  handle.param<double>("min_affine_precision", minPrecision, 1e-7);
  handle.param<double>("affine_scaling_factor", imageScaling, 4.0);
  return new AffineTransformEstimator(maxIterations,
                                      minPrecision,
                                      imageScaling);
}

VisualUtilityEstimator* CreateVisualUtilityEstimator(
  ros::NodeHandle handle,
  const TransformEstimator& transformEstimator) {

  double imgScaling;
  handle.param<double>("img_scaling", imgScaling, 1.0);

  string vuWrapperClass;
  handle.param<string>("vu_estimator_wrapper", vuWrapperClass, "");

  string vuEstimatorClass;
  handle.param<string>("vu_estimator", vuEstimatorClass,
                       "LABMotionVUEstimator");
  VisualUtilityEstimator* baseEstimator = NULL;
  if (vuEstimatorClass == "LABMotionVUEstimator") {
    double paretoThreshold;
    double distDecay;
    int openingSize;
    handle.param<double>("pareto_thresh", paretoThreshold, 0.03);
    handle.param<double>("dist_decay", distDecay, 2.0);
    handle.param<int>("opening_size", openingSize, 3);
    baseEstimator = new LABMotionVUEstimator(transformEstimator,
                                             paretoThreshold,
                                             distDecay,
                                             openingSize);
  } else if (vuEstimatorClass == "SpectralSaliency") {
    baseEstimator = new SpectralSaliency();
  } else if (vuEstimatorClass == "LaplacianVU") {
    int laplacianSize;
    handle.param<int>("laplacian_size", laplacianSize, 3);
    baseEstimator = new LaplacianVU(laplacianSize);
  } else if (vuEstimatorClass == "CenterSurroundHistogram") {
    string histDistType;
    handle.param<string>("hist_dist_type", histDistType, "chisq");
    string scaleStr;
    handle.param<string>("estimator_scales", scaleStr, "1.0");
    vector<double> estimatorScales = ParseDataList<double>(scaleStr);
    baseEstimator = new CenterSurroundHistogram(estimatorScales,
                                                histDistType);
  } else if (vuEstimatorClass == "Objectness") {
    baseEstimator = new Objectness();
  } else if (vuEstimatorClass == "HOGDetector") {
    string hogModelFile;
    handle.param<string>("hog_model_file", hogModelFile, "");
    bool hogDoPeople;
    handle.param<bool>("hog_do_people", hogDoPeople, true);
    bool hogDoCache;
    handle.param<bool>("hog_do_cache", hogDoCache, true);
    int winStride;
    cv::Size cvWinStride;
    if (handle.getParam("win_stride", winStride)) {
      if (vuWrapperClass == "ScaledDetectorWrapper") {
        winStride = cvRound(winStride*imgScaling);
      }
      cvWinStride = cv::Size(winStride, winStride);
    }
    baseEstimator = new HOGDetector(hogModelFile,
                                    hogDoPeople,
                                    cvWinStride,
                                    hogDoCache);
  } else if (vuEstimatorClass == "IntegralHOGDetector") {
    string hogModelFile;
    handle.param<string>("hog_model_file", hogModelFile, "");

    int winStride;
    cv::Size cvWinStride;
    handle.param<int>("win_stride", winStride, 8);
    cvWinStride = cv::Size(winStride, winStride);

    baseEstimator = new IntegralHOGDetector(hogModelFile, cvWinStride);
  } else if (vuEstimatorClass == "IntegralHOGCascade") {
    string hogModelFile;
    handle.param<string>("hog_model_file", hogModelFile, "");

    int winStride;
    cv::Size cvWinStride;
    handle.param<int>("win_stride", winStride, 8);
    cvWinStride = cv::Size(winStride, winStride);

    baseEstimator = new IntegralHOGCascade(hogModelFile, cvWinStride);
  } else if (vuEstimatorClass == "CascadeDetector") {
    string modelFile;
    handle.param<string>("cascade_model_file", modelFile, "");
    baseEstimator = new CascadeDetector(modelFile);
  } 

  if (vuWrapperClass == "RelativeEntropyVUWrapper") {
    return new RelativeEntropyVUWrapper(baseEstimator);
  } else if (vuWrapperClass == "ScaledDetectorWrapper") {
    return new ScaledDetectorWrapper(baseEstimator, imgScaling);
  } else {
    return baseEstimator;
  }

  ROS_FATAL("Invalid VisualUtilityEstimator specified. "
            "Please set the vu_estimator parameters");
  return NULL;

}

VisualUtilityMosaic* CreateVisualUtilityMosaic(
  ros::NodeHandle handle,
  const TransformEstimator& transformEstimator) {
  int morphCloseSize;
  double gaussSigma;
  handle.param<int>("morph_close_size", morphCloseSize, 0);
  handle.param<double>("gauss_sigma", gaussSigma, 0.0);
  return new NullVUMosaic(morphCloseSize, gaussSigma);
}

FrameEstimator* CreateFrameEstimator(ros::NodeHandle handle) {
  int xFramesize;
  int yFramesize;
  double frameExpansion;
  string frameEstimatorClass;
  handle.param<int>("xframesize", xFramesize, 192);
  handle.param<int>("yframesize", yFramesize, 108);
  handle.param<double>("frame_expansion", frameExpansion, 1.0);
  handle.param<string>("frame_estimator", frameEstimatorClass,
                            "MaxPointConstantFramesize");
  if (frameEstimatorClass == "MaxPointConstantFramesize") {
    return new MaxPointConstantFramesize(
      Size_<int>(xFramesize, yFramesize));
  } else if (frameEstimatorClass == "RandomPointConstantFramesize") {
    return new RandomPointConstantFramesize(
      Size_<int>(xFramesize, yFramesize));
  } else if (frameEstimatorClass == "HighRelativeEntropy") {
    int minFrameArea;
    double minEntropy;
    handle.param<int>("min_frame_area", minFrameArea, 200);
    handle.param<double>("min_entropy", minEntropy, 0.2);
    return new HighRelativeEntropy(frameExpansion,
                                   minFrameArea,
                                   minEntropy);
  }

  ROS_FATAL("Invalid FrameEstimator specified. "
            "Please set the frame_estimator parameters");
  return NULL;
}

} //namespace
