#include "hog_detector/hog_detector_internal.h"

#include <gtest/gtest.h>
#include <ros/ros.h>
#include <boost/scoped_ptr.hpp>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include "cv_bridge/cv_bridge.h"
#include <sensor_msgs/image_encodings.h>
#include <algorithm>
#include <math.h>
#include <vector>
#include "cv_utils/DisplayImages.h"

using namespace objdetect_msgs;
using namespace boost;
using namespace std;
using namespace cv;
namespace enc = sensor_msgs::image_encodings;

namespace hog_detector {

class HogDetectorTest : public ::testing::Test {
 protected:
  vector<Rect> foundLocations_;
  vector<double> scores_;
  scoped_ptr<HogDetectorImpl> detector_;

  virtual void SetUp() {
    foundLocations_.clear();
    scores_.clear();
    detector_.reset(new HogDetectorImpl());
    // Initialize the detector as a person detector with non-maximal supression
    detector_->InitModel("", true, 0.0, true);
  }

  // Returns true if the detection is in the array
  void ExpectDetectionInArray(const vector<Rect> arr,
                              const Rect& det) {
    for (vector<Rect>::const_iterator i = arr.begin();
         i != arr.end(); ++i) {
      if (i->x == det.x &&
          i->y == det.y &&
          i->width == det.width &&
          i->height == det.height) {
        return;
      }
    }
    ADD_FAILURE();    
  }

};

TEST_F(HogDetectorTest, EmptyImage) {
  ASSERT_TRUE(detector_->DetectObjects(Mat(),
                                       &foundLocations_,
                                       &scores_, NULL));

  EXPECT_EQ(foundLocations_.size(), 0u);
  EXPECT_EQ(scores_.size(), 0u);
}

TEST_F(HogDetectorTest, EmptyImageAndRegions) {
  ASSERT_TRUE(detector_->DetectObjects(Mat(),
                                       vector<Rect>(),
                                       &foundLocations_,
                                       &scores_, NULL));

  EXPECT_EQ(foundLocations_.size(), 0u);
  EXPECT_EQ(scores_.size(), 0u);
}

TEST_F(HogDetectorTest, ManualWindowsSameAsAutomatic) {
  Mat cvImage = imread("test/I00000.png");

  double processingTime;

  // First find the people in the image using a multiple scale detector
  detector_->DetectObjects(cvImage, &foundLocations_,
                           &scores_, &processingTime);

  ROS_INFO_STREAM("Processing time is: " << processingTime);

  ASSERT_GT(foundLocations_.size(), 0u);

  // Save a copy of the detections
  vector<Rect> multiScaleDetections = foundLocations_;
  vector<double> multiScaleScores = scores_;
  foundLocations_.clear();
  scores_.clear();

  // Draw the rectangles from the first detection
  Mat multiImage = cvImage.clone();
  for (vector<Rect>::const_iterator i = multiScaleDetections.begin();
       i != multiScaleDetections.end(); ++i) {
    rectangle(multiImage, Point(i->x, i->y),
              Point(i->x + i->width,
                    i->y + i->height),
              CV_RGB(255, 0, 0));
  }
  cv_utils::DisplayNormalizedImage(multiImage, "multiImage");

  // Send the request
  detector_->InitModel("", true, -110.0, true);
  detector_->DetectObjects(cvImage, multiScaleDetections,
                           &foundLocations_,
                           &scores_, NULL);

  Mat manualImage = cvImage.clone();
  for (vector<Rect>::const_iterator i = foundLocations_.begin();
       i != foundLocations_.end(); ++i) {
    rectangle(manualImage, Point(i->x, i->y),
              Point(i->x + i->width,
                    i->y + i->height),
              CV_RGB(0, 0, 255));
  }
  cv_utils::DisplayNormalizedImage(manualImage, "manualImage");
  cv_utils::ShowWindowsUntilKeyPress();

  // Make sure that the two responses are identical
  ASSERT_EQ(multiScaleDetections.size(), foundLocations_.size());
  for (vector<Rect>::const_iterator i = multiScaleDetections.begin();
       i != multiScaleDetections.end(); ++i) {
    ExpectDetectionInArray(foundLocations_, *i);
  }
  for (unsigned int i = 0u; i < scores_.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(scores_[i], multiScaleScores[i]);
  }
}


} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ros::Time::init();
  return RUN_ALL_TESTS();
}
