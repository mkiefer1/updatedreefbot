#include "hog_detector/interpolation.h"

#include <gtest/gtest.h>
#include <ros/ros.h>
#include <vector>
#include <boost/scoped_ptr.hpp>

using namespace std;
using namespace boost;

class LinearInterpTest : public ::testing::Test {
 protected:
  vector<double> x_;
  vector<double> y_;

  virtual void SetUp() {
    x_.clear();
    y_.clear();
  }
};

TEST_F(LinearInterpTest, EmptyInterpolator) {
  ASSERT_EQ(x_.size(), 0);
  ASSERT_EQ(y_.size(), 0);
  EXPECT_DEATH(LinearInterpolator(x_, y_),
               ".* insufficient number of points for interpolation type");
}

TEST_F(LinearInterpTest, SortedInterpolation) {
  double x[] = {0, 3, 4, 8, 9, 15, 17};
  double y[] = {1, 6, 9, 23, 15, 14, 18};
  x_.insert(x_.end(), x, x+7);
  y_.insert(y_.end(), y, y+7);
  LinearInterpolator interpolator(x_, y_);

  EXPECT_FLOAT_EQ(interpolator(2.0), 1 + 2.0*5/3);
  EXPECT_FLOAT_EQ(interpolator(3.5), 7.5);
  EXPECT_FLOAT_EQ(interpolator(10), 15 - 1.0/6);
  EXPECT_FLOAT_EQ(interpolator(14), 15 - 5.0/6);
  EXPECT_FLOAT_EQ(interpolator(16), 16.0);
}

TEST_F(LinearInterpTest, UnsortedInterpolation) {
  double x[] = {15, 9, 17, 0, 3, 4, 8};
  double y[] = {14, 15, 18, 1, 6, 9, 23};
  x_.insert(x_.end(), x, x+7);
  y_.insert(y_.end(), y, y+7);
  LinearInterpolator interpolator(x_, y_);

  EXPECT_FLOAT_EQ(interpolator(2.0), 1 + 2.0*5/3);
  EXPECT_FLOAT_EQ(interpolator(3.5), 7.5);
  EXPECT_FLOAT_EQ(interpolator(10), 15 - 1.0/6);
  EXPECT_FLOAT_EQ(interpolator(14), 15 - 5.0/6);
  EXPECT_FLOAT_EQ(interpolator(16), 16.0);
}

TEST_F(LinearInterpTest, IdenticalEntryInterpolation) {
  double x[] = {0, 3, 3, 4, 8, 9, 15, 17};
  double y[] = {1, 6, 6, 9, 23, 15, 14, 18};
  x_.insert(x_.end(), x, x+8);
  y_.insert(y_.end(), y, y+8);
  LinearInterpolator interpolator(x_, y_);

  EXPECT_FLOAT_EQ(interpolator(2.0), 1 + 2.0*5/3);
  EXPECT_FLOAT_EQ(interpolator(3.5), 7.5);
  EXPECT_FLOAT_EQ(interpolator(10), 15 - 1.0/6);
  EXPECT_FLOAT_EQ(interpolator(14), 15 - 5.0/6);
  EXPECT_FLOAT_EQ(interpolator(16), 16.0);
}

TEST_F(LinearInterpTest, OutOfBounds) {
  double x[] = {0, 3, 4, 8, 9, 15, 17};
  double y[] = {1, 6, 9, 23, 15, 14, 18};
  x_.insert(x_.end(), x, x+7);
  y_.insert(y_.end(), y, y+7);
  LinearInterpolator interpolator(x_, y_);

  EXPECT_THROW(interpolator(-3.0), Interpolator::out_of_bounds);
  EXPECT_THROW(interpolator(22.0), Interpolator::out_of_bounds);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ros::Time::init();
  return RUN_ALL_TESTS();
}
