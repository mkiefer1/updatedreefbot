#include <gtest/gtest.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cascade_parts_detector/cascade_parts_detector.h"

#include "sensor_msgs/Image.h"
#include "cascade_parts_detector/DetectionArray.h"
#include "cascade_parts_detector/DetectObject.h"
#include "cv_bridge/cv_bridge.h"

#ifndef ROOT_DIR
#define ROOT_DIR "."
#endif

using namespace std;
using namespace cv;
using sensor_msgs::Image;
using cv_bridge::CvImage;

template<typename T>
T min(const T& a, const T& b) { return a < b ? a : b; }

namespace cascade_parts_detector {

class CascadePartsDetectorTest : public ::testing::Test {
protected:
  CascadePartsDetector detector_;
  Mat image_;

  virtual void SetUp() {
    detector_.modelFile_ = string(ROOT_DIR) + 
      "/src/voc_release4/VOC2009/person_final.mat";
    detector_.thresh_ = "0";
    detector_.doCascade_ = true;
    detector_.doTiming_ = false;
    ASSERT_TRUE(detector_.InitMatlab());

    string imagePath = string(ROOT_DIR) + "/test/img_0030.bmp";
    image_ = imread(imagePath);
  }

  void HandleRequest(const Mat& image,
                     DetectionArray* response) {
    CvImage cvImage;
    cvImage.image = image;
    cvImage.encoding = "bgr8";
    Image::Ptr msg = cvImage.toImageMsg();
    ASSERT_TRUE(detector_.HandleRequestImpl(*msg, response));
  }

};

TEST_F(CascadePartsDetectorTest, FindPersonInWholeImage) {
  DetectionArray response;
  HandleRequest(image_, &response);

  ASSERT_EQ(response.detections.size(), 1u);
}

TEST_F(CascadePartsDetectorTest, FrameFractionTest) {
  // Makes sure that sending a smaller frame, which contains the
  // person, still gives finds that person.
  DetectionArray fullResponse;
  HandleRequest(image_, &fullResponse);

  ASSERT_EQ(fullResponse.detections.size(), 1u);

  // Calculate the window of the image to send
  int xFramesize = image_.cols * 0.9;
  int yFramesize = image_.rows * 0.9;
  Rect roi(std::max<int>(0,
             std::min<int>(image_.cols - xFramesize, 
                           fullResponse.detections[0].x_offset + 
                           fullResponse.detections[0].width/2.0 -
                           xFramesize / 2.0)),
           std::max<int>(0,
             std::min<int>(image_.rows - yFramesize, 
                           fullResponse.detections[0].y_offset + 
                           fullResponse.detections[0].height/2.0 -
                           yFramesize / 2.0)),
           xFramesize,
           yFramesize);
  Mat smallImage(image_, roi);

  DetectionArray smallResponse;
  HandleRequest(smallImage, &smallResponse);
  ASSERT_EQ(smallResponse.detections.size(), 1u);

  EXPECT_EQ(smallResponse.detections[0].width,
            fullResponse.detections[0].width);
  EXPECT_EQ(smallResponse.detections[0].height,
            fullResponse.detections[0].height);
  EXPECT_EQ(smallResponse.detections[0].x_offset,
            fullResponse.detections[0].x_offset - roi.x);
  EXPECT_EQ(smallResponse.detections[0].y_offset,
            fullResponse.detections[0].y_offset - roi.y);
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
