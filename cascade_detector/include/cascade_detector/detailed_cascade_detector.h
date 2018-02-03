// An extension of the OpenCV cascade detector that allows you to
// query for specific windows in the image.
//
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
#ifndef __DETAILED_CASCADE_DETECTOR_H__
#define __DETAILED_CASCADE_DETECTOR_H__

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <ros/ros.h>
#include <string.h>
#include <vector>

namespace cascade_detector {

class DetailedCascadeDetector : public cv::CascadeClassifier {
public:
  DetailedCascadeDetector() : cv::CascadeClassifier() {}
  virtual ~DetailedCascadeDetector();

  // Load up the model
  bool Init(const std::string& filename);

  // Detect objects in an image
  // Inputs:
  // image - image to look for objects in
  // roisIn - rectangles to evaluate
  //
  // Outputs:
  // foundLocations - rectangles where the objects were found
  // scores - score for each of the input rois. It is equal to the stage 
  //          that the frame was thrown out on using 0 based indexing. 
  //          It is equal to GetNumCascadeStages if the object was found.
  //          (optional)
  // processingTime - Processing time to find the objects (optional)
  // integralTime - Processing time to compute the integral (optional)
  bool DetectObjects(const cv::Mat& image,
                     const std::vector<cv::Rect>& roisIn,
                     std::vector<cv::Rect>* foundLocations,
                     std::vector<int>* scores,
                     double* processingTime,
                     double* integralTime=NULL); 

  // Returns the number of stages in the cascade
  int GetNumCascadeStages() const {
    if (isOldFormatCascade()) {
      return oldCascade->count;
    }
    return this->data.stages.size();
  }

  // Retrieve the window size assumed in the cascade. It can handle
  // the old cascade format properly.
  cv::Size GetWindowSize() const {
    if (isOldFormatCascade()) {
      return oldCascade->orig_window_size;
    }
    return getOriginalWindowSize();
  }

private:
};

} // namespace

#endif // __DETAILED_CASCADE_DETECTOR_H__
