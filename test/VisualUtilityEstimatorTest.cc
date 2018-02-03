// Tests for those visual utility estimators that you can do good
// unittests for.

#include <boost/scoped_ptr.hpp>
#include <gtest/gtest.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "visual_utility/VisualUtilityEstimator.h"

using namespace cv;
using namespace std;
using namespace boost;

namespace visual_utility {

class CenterSurroundHistogramTest : public ::testing::Test {
 protected:
  scoped_ptr<VisualUtilityEstimator> estimator;

  virtual void SetUp() {
    vector<double> scales;
    scales.push_back(1.0);
    estimator.reset(new CenterSurroundHistogram(scales, "chisq"));
  }

};

TEST_F(CenterSurroundHistogramTest, ZeroesAndOnesSeperate) {
  Mat_<uint8_t> image = Mat_<uint8_t>::zeros(32, 32);
  image(Rect(14,14,4,4)) = 255;

  vector<Rect> rois;
  rois.push_back(Rect(14, 14, 4, 4));

  vector<VisualUtilityEstimator::ROIScore> scores;
  estimator->CalculateVisualUtility(image, rois, 0, &scores);

  ASSERT_EQ(scores.size(), 1u);
  EXPECT_EQ(scores[0].second.x, 14);
  EXPECT_EQ(scores[0].second.y, 14);
  EXPECT_EQ(scores[0].second.width, 4);
  EXPECT_EQ(scores[0].second.height, 4);
  EXPECT_FLOAT_EQ(scores[0].first, 2.0);
}

TEST_F(CenterSurroundHistogramTest, ZeroesAndOnesOverlap) {
  Mat_<uint8_t> image = Mat_<uint8_t>::zeros(32, 32);
  image(Rect(15,14,4,4)) = 255;

  vector<Rect> rois;
  rois.push_back(Rect(14, 14, 4, 4));

  vector<VisualUtilityEstimator::ROIScore> scores;
  estimator->CalculateVisualUtility(image, rois, 0, &scores);

  ASSERT_EQ(scores.size(), 1u);
  EXPECT_EQ(scores[0].second.x, 14);
  EXPECT_EQ(scores[0].second.y, 14);
  EXPECT_EQ(scores[0].second.width, 4);
  EXPECT_EQ(scores[0].second.height, 4);
  EXPECT_FLOAT_EQ(scores[0].first, 0.6065163);
}

class LaplacianEntropyTest : public ::testing::Test {
 protected:
  scoped_ptr<VisualUtilityEstimator> estimator;

  virtual void SetUp() {
    estimator.reset(new RelativeEntropyVUWrapper(new LaplacianVU(5)));
  }

};

TEST_F(LaplacianEntropyTest, NanBox) {
  Mat image = imread("test/img_0003.png");

  vector<Rect> rois;
  rois.push_back(Rect(110, 200, 64, 128));

  vector<VisualUtilityEstimator::ROIScore> scores;
  estimator->CalculateVisualUtility(image, rois, 0, &scores);

  ROS_INFO_STREAM("Score is " << scores[0].first);
}

} // namespace


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ros::Time::init();
  return RUN_ALL_TESTS();
}
