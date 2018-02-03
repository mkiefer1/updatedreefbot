#include "hog_detector/integral_hog_detector.h"

#include <gtest/gtest.h>
#include <ros/ros.h>
#include <string>

using namespace std;
using namespace cv;

namespace hog_detector {

// Test to make sure that saving the detector to disk and then reading
// it back works properly.
TEST(IntegralHogDetectorTest, SaveLoadTest) {
  string saveFilename("test/integral_hog_detector_test.xml");

  IntegralHogDetector orig(Size(42, 46),
                           Size(8, 6),
                           Size(2, 1),
                           Size(4, 3),
                           7,
                           -5.0,
                           Rect(20, 21, 16, 12));

  orig.save(saveFilename);

  IntegralHogDetector copy(saveFilename, Size());
  
  EXPECT_EQ(orig.winSize().width, copy.winSize().width);
  EXPECT_EQ(orig.winSize().height, copy.winSize().height);
  EXPECT_EQ(orig.subWindow().width, copy.subWindow().width);
  EXPECT_EQ(orig.subWindow().height, copy.subWindow().height);
  EXPECT_EQ(orig.subWindow().x, copy.subWindow().x);
  EXPECT_EQ(orig.subWindow().y, copy.subWindow().y);
  EXPECT_FLOAT_EQ(orig.thresh(), copy.thresh());
  EXPECT_EQ(orig.nbins(), copy.nbins());
  EXPECT_EQ(orig.generator().winSize().width,
            copy.generator().winSize().width);
  EXPECT_EQ(orig.generator().winSize().height,
            copy.generator().winSize().height);
  EXPECT_EQ(orig.generator().blockSize().width,
            copy.generator().blockSize().width);
  EXPECT_EQ(orig.generator().blockSize().height,
            copy.generator().blockSize().height);
  EXPECT_EQ(orig.generator().blockStride().width,
            copy.generator().blockStride().width);
  EXPECT_EQ(orig.generator().blockStride().height,
            copy.generator().blockStride().height);
  EXPECT_EQ(orig.generator().cellSize().width,
            copy.generator().cellSize().width);
  EXPECT_EQ(orig.generator().cellSize().height,
            copy.generator().cellSize().height);
  EXPECT_EQ(orig.generator().nbins(), copy.generator().nbins());
  EXPECT_EQ(orig.generator().descriptorSize(),
            copy.generator().descriptorSize());
}

}// namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ros::Time::init();
  return RUN_ALL_TESTS();
}
