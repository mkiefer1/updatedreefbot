#include "visual_utility/GeometricTransforms-Inl.h"
#include <gtest/gtest.h>

using namespace cv;
using namespace std;

namespace geo {

class BilinearInterpTest : public ::testing::Test {
 protected:
  Mat_<double> image;

  virtual void SetUp() {
    // Create an image that is 0 1 2 3 4 5
    image = (Mat_<double>(2,3) << 0, 1, 2, 3, 4, 5);
  }
};

TEST_F(BilinearInterpTest, TestCornerInterp) {
  // Test interpolation on the corners
  double interpVal;

  BilinearInterpPoint(image, 0, 0, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 0);

  BilinearInterpPoint(image, 0, 1, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 3);

  BilinearInterpPoint(image, 1, 0, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 1);

  BilinearInterpPoint(image, 1, 1, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 4);

  BilinearInterpPoint(image, 2, 0, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 2);

  BilinearInterpPoint(image, 2, 1, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 5);
}

TEST_F(BilinearInterpTest, TestEdgeInterp) {
  // Test interpolation on the corners
  double interpVal;

  BilinearInterpPoint(image, 0.5, 0, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 0.5);

  BilinearInterpPoint(image, 0.75, 0, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 0.75);

  BilinearInterpPoint(image, 0, 0.5, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 1.5);

  BilinearInterpPoint(image, 0, 0.25, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 0.75);

  BilinearInterpPoint(image, 1, 0.75, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 3.25);

  BilinearInterpPoint(image, 1, 0.5, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 2.5);

  BilinearInterpPoint(image, 1.5, 1, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 4.5);

  BilinearInterpPoint(image, 1.25, 1, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 4.25);
}

TEST_F(BilinearInterpTest, TestInnerInterp) {
  // Test interpolation on the corners
  double interpVal;

  BilinearInterpPoint(image, 0.5, 0.5, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 2.0);
  BilinearInterpPoint(image, 0.5, 0.25, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 1.25);
  BilinearInterpPoint(image, 0.5, 0.75, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 2.75);

  BilinearInterpPoint(image, 1.5, 0.5, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 3.0);
  BilinearInterpPoint(image, 1.5, 0.25, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 2.25);
  BilinearInterpPoint(image, 1.5, 0.75, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 3.75);

  BilinearInterpPoint(image, 1.25, 0.5, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 2.75);
  BilinearInterpPoint(image, 1.75, 0.5, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 3.25);

  
  BilinearInterpPoint(image, 1.75, 0.75, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal, 4.0);

}

TEST_F(BilinearInterpTest, MultiChannelImage) {
  // Create an image to interpolate in
  Mat_<Vec2d> multiImage = (Mat_<Vec2d>(2,2) <<
                            Vec2d(0, 10),
                            Vec2d(1, 11),
                            Vec2d(2, 12),
                            Vec2d(3, 13));
  
  // Test interpolation on the corners
  Vec2d interpVal;

  BilinearInterpPoint(multiImage, 0.5, 0.5, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal[0], 1.5);
  EXPECT_DOUBLE_EQ(interpVal[1], 11.5);

  BilinearInterpPoint(multiImage, 1, 0, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal[0], 1);
  EXPECT_DOUBLE_EQ(interpVal[1], 11);

  BilinearInterpPoint(multiImage, 0, 0.25, &interpVal);
  EXPECT_DOUBLE_EQ(interpVal[0], 0.5);
  EXPECT_DOUBLE_EQ(interpVal[1], 10.5);
}


class AffineTransformTest : public ::testing::Test {
 protected:
  Mat_<double> image;
  Mat_<double> M;

  virtual void SetUp() {
    // Create an image that is zeros except for some single pixels of value 1
    image = Mat_<double>::zeros(15,20);
    // x = 5, y = 10
    image(10, 5) = 1.0;

    // x = 6, y = 3
    image(3, 6) = 1.0;

    // Create an identity matrix for the affine transform
    M = Mat_<double>::zeros(2, 3);
    M(0, 0) = 1;
    M(1, 1) = 1;
  }
};

TEST_F(AffineTransformTest, TestIdentityTransform) {
  Mat_<double> warpedImage = AffineWarp(image, M, geo::BORDER_CONSTANT,
                                        0.0, image);

  
  EXPECT_DOUBLE_EQ(warpedImage(3,6), 1.0);
  EXPECT_DOUBLE_EQ(warpedImage(10,5), 1.0);
  EXPECT_DOUBLE_EQ(warpedImage(10,6), 0);
  EXPECT_DOUBLE_EQ(warpedImage(10,4), 0);
  EXPECT_DOUBLE_EQ(warpedImage(11,5), 0);
  EXPECT_DOUBLE_EQ(warpedImage(9,5), 0);
}

TEST_F(AffineTransformTest, TestTranslationTransform) {
  M(0,2) = 2; // X difference
  M(1,2) = -3; // Y difference
  Mat_<double> warpedImage = AffineWarp(image, M, geo::BORDER_CONSTANT,
                                        0.0);

  
  EXPECT_DOUBLE_EQ(warpedImage(0,8), 1.0);
  EXPECT_DOUBLE_EQ(warpedImage(7,7), 1.0);
  EXPECT_DOUBLE_EQ(warpedImage(7,8), 0);
  EXPECT_DOUBLE_EQ(warpedImage(7,6), 0);
  EXPECT_DOUBLE_EQ(warpedImage(8,7), 0);
  EXPECT_DOUBLE_EQ(warpedImage(6,7), 0);
}

TEST_F(AffineTransformTest, TestNonIntegerTranslationTransform) {
  M(0,2) = 2.5; // X difference
  M(1,2) = -3; // Y difference
  Mat_<double> warpedImage = AffineWarp(image, M, geo::BORDER_CONSTANT,
                                        0.0);

  
  EXPECT_DOUBLE_EQ(warpedImage(0,8), 0.5);
  EXPECT_DOUBLE_EQ(warpedImage(7,7), 0.5);
  EXPECT_DOUBLE_EQ(warpedImage(7,8), 0.5);
  EXPECT_DOUBLE_EQ(warpedImage(7,6), 0);
  EXPECT_DOUBLE_EQ(warpedImage(8,7), 0);
  EXPECT_DOUBLE_EQ(warpedImage(6,7), 0);
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
