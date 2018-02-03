#include "visual_utility/FrameEstimator.h"
#include "visual_utility/VisualUtilityMosaic.h"

#include <gtest/gtest.h>
#include <boost/scoped_ptr.hpp>

using namespace cv;
using namespace std;
using namespace boost;
using namespace cv_blobs;

namespace visual_utility {

class MaxPointConstantFramesizeTest : public ::testing::Test {
 protected:
  scoped_ptr<VisualUtilityMosaic> mosaic;
  scoped_ptr<FrameEstimator> frameEstimator;
  vector<FrameOfInterest> frames;

  virtual void SetUp() {
    mosaic.reset(new NullVUMosaic(0,0));
    frameEstimator.reset(new MaxPointConstantFramesize(Size_<int>(3,2)));
    frames.clear();
  }
  
};

TEST_F(MaxPointConstantFramesizeTest, SingleMaxPointInCenter) {
  Mat_<double> frame = Mat_<double>::ones(6,8);
  frame[2][4] = 10;
  frame[0][5] = -16;
  mosaic->AddFrame(frame, NULL);

  frameEstimator->FindInterestingFrames(*mosaic, &frames);
  ASSERT_EQ(frames.size(), 1u);
  EXPECT_FLOAT_EQ(frames[0].height, 2);
  EXPECT_FLOAT_EQ(frames[0].width, 3);
  EXPECT_FLOAT_EQ(frames[0].xCenter, 0);
  EXPECT_FLOAT_EQ(frames[0].yCenter, -1);

}

TEST_F(MaxPointConstantFramesizeTest, SingleMaxPointOnLeftEdge) {
  Mat_<double> frame = Mat_<double>::ones(6,8);
  frame[2][0] = 10;
  frame[0][5] = -16;
  mosaic->AddFrame(frame, NULL);

  frameEstimator->FindInterestingFrames(*mosaic, &frames);
  ASSERT_EQ(frames.size(), 1u);
  EXPECT_FLOAT_EQ(frames[0].height, 2);
  EXPECT_FLOAT_EQ(frames[0].width, 3);
  EXPECT_FLOAT_EQ(frames[0].xCenter, -4);
  EXPECT_FLOAT_EQ(frames[0].yCenter, -1);

}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
