#include "visual_utility/VisualUtilityMosaic.h"

#include <gtest/gtest.h>
#include <boost/scoped_ptr.hpp>

using namespace cv;
using namespace std;
using namespace boost;
using namespace cv_blobs;

namespace visual_utility {

class NullVUMosaicTest : public ::testing::Test {
 protected:
  scoped_ptr<NullVUMosaic> mosaic;

  virtual void SetUp() {
    mosaic.reset(new NullVUMosaic(0, 0));
  }

};

TEST_F(NullVUMosaicTest, IntegralTest) {
  Mat_<double> frame = Mat_<double>::ones(4,6);
  mosaic->AddFrame(frame, NULL);

  EXPECT_DOUBLE_EQ(mosaic->GetIntegralValue(Point2f(-4, -4)), 0);
  EXPECT_DOUBLE_EQ(mosaic->GetIntegralValue(Point2f(4, 4)), 24);
  EXPECT_DOUBLE_EQ(mosaic->GetIntegralValue(Point2f(0, 0)), 6);
  EXPECT_DOUBLE_EQ(mosaic->GetIntegralValue(Point2f(-3, 0)), 0);
  EXPECT_DOUBLE_EQ(mosaic->GetIntegralValue(Point2f(0, -4)), 0);
}

TEST_F(NullVUMosaicTest, SizeTest) {
  EXPECT_DOUBLE_EQ(mosaic->GetSize(), 0);

  Mat_<double> frame = Mat_<double>::ones(4,6);
  mosaic->AddFrame(frame, NULL);
  
  EXPECT_DOUBLE_EQ(mosaic->GetSize(), 24);
}

TEST_F(NullVUMosaicTest, SumTest) {
  EXPECT_DOUBLE_EQ(mosaic->GetSum(), 0);

  Mat_<double> frame = Mat_<double>::ones(4,6);
  mosaic->AddFrame(frame, NULL);
  EXPECT_DOUBLE_EQ(mosaic->GetSum(), 24);

  frame[2][3] = 10;
  mosaic->AddFrame(frame, NULL);
  EXPECT_DOUBLE_EQ(mosaic->GetSum(), 33);
}

TEST_F(NullVUMosaicTest, ValueTest) {
  EXPECT_DOUBLE_EQ(mosaic->GetValue(Point2f(0, 0)), 0);

  Mat_<double> frame = Mat_<double>::ones(4,6);
  frame[2][3] = 10;
  frame[0][5] = -16;
  mosaic->AddFrame(frame, NULL);
  EXPECT_DOUBLE_EQ(mosaic->GetValue(Point2f(0, 0)), 10);
  EXPECT_DOUBLE_EQ(mosaic->GetValue(Point2f(2, -2)), -16);
  EXPECT_DOUBLE_EQ(mosaic->GetValue(Point2f(4, 4)), 0);
  EXPECT_DOUBLE_EQ(mosaic->GetValue(Point2f(-4, -4)), 0);
}

struct MaxPixelFunc : public PixelValFunc {
  virtual double operator()(const double& val) { return val; };
};

TEST_F(NullVUMosaicTest, MaximumPixelTest) {
  Mat_<double> frame = Mat_<double>::ones(4,6);
  frame[1][3] = 10;
  frame[0][5] = -16;
  mosaic->AddFrame(frame, NULL);

  Point2f maxLoc;
  EXPECT_DOUBLE_EQ(mosaic->MaximizeFunction(&MaxPixelFunc(), &maxLoc), 10);
  EXPECT_DOUBLE_EQ(maxLoc.x, 0);
  EXPECT_DOUBLE_EQ(maxLoc.y, -1);
  EXPECT_DOUBLE_EQ(mosaic->GetValue(maxLoc), 10);
}

TEST_F(NullVUMosaicTest, FindConnectedComponents) {
  Mat_<double> frame = Mat_<double>::ones(4,6);
  frame[1][3] = 10;
  frame[0][3] = 10;
  frame[2][0] = 9;
  frame[0][5] = -16;
  mosaic->AddFrame(frame, NULL);

  BlobResult<double> blobs;
  mosaic->FindConnectedComponents(7, &blobs);
  ASSERT_EQ(blobs.nBlobs(), 2);
  for (int i = 0; i < blobs.nBlobs(); i++) {
    const Blob& curBlob = blobs.GetBlob(i);
    if (curBlob.area() == 1) {
      EXPECT_EQ(curBlob.minX(), -3);
      EXPECT_EQ(curBlob.maxX(), -3);
      EXPECT_EQ(curBlob.minY(), 0);
      EXPECT_EQ(curBlob.maxY(), 0);
    } else if (curBlob.area() == 2) {
      EXPECT_EQ(curBlob.minX(), 0);
      EXPECT_EQ(curBlob.maxX(), 0);
      EXPECT_EQ(curBlob.minY(), -2);
      EXPECT_EQ(curBlob.maxY(), -1);
    } else {
      FAIL() << "We shouldn't have found a blob of size: " << curBlob.area();
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
