// A ROS and OpenCVwrapper for the objectness code from 
//
// Rahtu E. & Kannala J. & Blaschko M. B. 
// Learning a Category Independent Object Detection Cascade. 
// Proc. International Conference on Computer Vision (ICCV 2011).
//
// Author: Mark Desnoyer (markd@cmu.edu)
// Date: March 2012
#ifndef __OBJECTNESS_OBJECTNESS_H__
#define __OBJECTNESS_OBJECTNESS_H__

#include <opencv2/core/core.hpp>
#include <vector>
#include <ros/ros.h>
#include <string>

namespace objectness {

class Objectness {
public:
  Objectness(bool doTiming)
    : doTiming_(doTiming) {}
  ~Objectness();

  typedef std::pair<double, cv::Rect> ROIScore;

  bool Init();

  // Calculate objectness for a bunch of regions of interest with the
  // data being in OpenCV format.
  void CalculateObjectness(const cv::Mat& image,
                           const std::vector<cv::Rect>& rois,
                           std::vector<ROIScore>* scoreOut,
                           double* runtime);

private:
  bool doTiming_;
  char matlabBuffer_[1024];

  // Initializes the matlab engine and the variables needed to serve data
  bool InitMatlab();

  // Helper function that finds the directory where this package is
  // running from
  std::string FindPackageDir() const;

};

} // namespace

#endif // __OBJECTNESS_OBJECTNESS_H__
