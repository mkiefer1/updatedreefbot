#include "cv_blobs/Blob.h"
#include <boost/scoped_ptr.hpp>
#include <gtest/gtest.h>
#include "opencv2/core/core.hpp"
#include <sstream>

#include "cv_bridge/CvBridge.h"

using namespace cv;
using namespace boost;
using namespace std;
using reefbot_msgs::ImageRegion;

namespace cv_blobs {

class BlobTest : public ::testing::Test {
 protected:
  scoped_ptr<Blob> blob;

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  virtual void SetUp() {
    blob.reset(new Blob(3, Blob::EIGHT_CONNECTED));
  }
};

TEST_F(BlobTest, EmptyInitially) {
  EXPECT_EQ(blob->id(), 3);
  EXPECT_EQ(blob->area(), 0);
}

TEST_F(BlobTest, OnePointAdded) {
  ASSERT_TRUE(blob->AddPoint(Point2i(5, 3)));
  ASSERT_EQ(blob->area(), 1);
  EXPECT_EQ(blob->minX(), 5);
  EXPECT_EQ(blob->maxX(), 5);
  EXPECT_EQ(blob->minY(), 3);
  EXPECT_EQ(blob->maxY(), 3);
}

TEST_F(BlobTest, OnePointAddedTwice) {
  ASSERT_TRUE(blob->AddPoint(Point2i(5, 3)));
  ASSERT_FALSE(blob->AddPoint(Point2i(5, 3)));
  ASSERT_EQ(blob->area(), 1);
  EXPECT_EQ(blob->minX(), 5);
  EXPECT_EQ(blob->maxX(), 5);
  EXPECT_EQ(blob->minY(), 3);
  EXPECT_EQ(blob->maxY(), 3);
}

TEST_F(BlobTest, TriangleAdded) {
  ASSERT_TRUE(blob->AddPoint(Point2i(5, 3)));
  ASSERT_TRUE(blob->AddPoint(Point2i(1, 3)));
  ASSERT_TRUE(blob->AddPoint(Point2i(2, 8)));
  ASSERT_EQ(blob->area(), 3);
  EXPECT_EQ(blob->minX(), 1);
  EXPECT_EQ(blob->maxX(), 5);
  EXPECT_EQ(blob->minY(), 3);
  EXPECT_EQ(blob->maxY(), 8);
}

class ToImageRegionTest : public ::testing::Test {
 protected:
  scoped_ptr<Blob> blob;

  virtual void SetUp() {
    blob.reset(new Blob(3, Blob::EIGHT_CONNECTED));
  }

  
};

TEST_F(ToImageRegionTest, FullBoundingBox) {
  ASSERT_TRUE(blob->AddPoint(Point(2,4)));
  ASSERT_TRUE(blob->AddPoint(Point(2,5)));
  ASSERT_TRUE(blob->AddPoint(Point(3,5)));
  ASSERT_TRUE(blob->AddPoint(Point(3,4)));
  ASSERT_TRUE(blob->AddPoint(Point(4,5)));
  ASSERT_TRUE(blob->AddPoint(Point(4,4)));

  ImageRegion::Ptr imageRegion = blob->ToImageRegion();

  ASSERT_EQ(imageRegion->bounding_box.x_offset, 2u);
  ASSERT_EQ(imageRegion->bounding_box.y_offset, 4u);
  ASSERT_EQ(imageRegion->bounding_box.height, 2u);
  ASSERT_EQ(imageRegion->bounding_box.width, 3u);
}

TEST_F(ToImageRegionTest, PartialBox) {
  ASSERT_TRUE(blob->AddPoint(Point(2,4)));
  ASSERT_TRUE(blob->AddPoint(Point(3,5)));
  ASSERT_TRUE(blob->AddPoint(Point(4,5)));

  ImageRegion::Ptr imageRegion = blob->ToImageRegion();

  ASSERT_EQ(imageRegion->bounding_box.x_offset, 2u);
  ASSERT_EQ(imageRegion->bounding_box.y_offset, 4u);
  ASSERT_EQ(imageRegion->bounding_box.height, 2u);
  ASSERT_EQ(imageRegion->bounding_box.width, 3u);

  ASSERT_EQ(imageRegion->mask.height, 2u);
  ASSERT_EQ(imageRegion->mask.width, 3u);
  ASSERT_EQ(imageRegion->mask.data[0], 1);
  ASSERT_EQ(imageRegion->mask.data[1], 0);
  ASSERT_EQ(imageRegion->mask.data[2], 0);
  ASSERT_EQ(imageRegion->mask.data[3], 0);
  ASSERT_EQ(imageRegion->mask.data[4], 1);
  ASSERT_EQ(imageRegion->mask.data[5], 1);
}

TEST_F(ToImageRegionTest, WithBridge) {
  ASSERT_TRUE(blob->AddPoint(Point(2,4)));
  ASSERT_TRUE(blob->AddPoint(Point(3,5)));
  ASSERT_TRUE(blob->AddPoint(Point(4,5)));

  ImageRegion::Ptr imageRegion = blob->ToImageRegion();
  sensor_msgs::CvBridge bridge;
  IplImage* maskImagePtr = NULL;
   sensor_msgs::Image::ConstPtr maskPtr(
     new sensor_msgs::Image(imageRegion->mask));
  maskImagePtr = bridge.imgMsgToCv(maskPtr, "mono8");
  Mat maskImage(maskImagePtr);

  ASSERT_EQ(maskImage.rows, 2);
  ASSERT_EQ(maskImage.cols, 3);
  ASSERT_EQ(maskImage.at<uchar>(0, 0), 1);
  ASSERT_EQ(maskImage.at<uchar>(0, 1), 0);
  ASSERT_EQ(maskImage.at<uchar>(1, 0), 0);
  ASSERT_EQ(maskImage.at<uchar>(1, 1), 1);
  ASSERT_EQ(maskImage.at<uchar>(0, 2), 0);
  ASSERT_EQ(maskImage.at<uchar>(1, 2), 1);
}

class SerializeTest : public ::testing::Test {
 protected:
  scoped_ptr<Blob> blob;
  scoped_ptr<stringstream> buf;

  virtual void SetUp() {
    blob.reset(new Blob(3, Blob::EIGHT_CONNECTED));
    ASSERT_TRUE(blob->AddPoint(Point2i(5,3)));
    ASSERT_TRUE(blob->AddPoint(Point2i(3,2)));
    buf.reset(new stringstream());
  }

};

TEST_F(SerializeTest, RoundTripSerialization) {
  blob->AsciiSerialize(*buf);
  
  shared_ptr<Blob> newBlob = Blob::CreateFromStream(*buf, 3);

  ASSERT_EQ(*blob, *newBlob);
}

TEST_F(SerializeTest, RoundTripEmptySerialization) {
  Blob curBlob(4, Blob::EIGHT_CONNECTED);
  curBlob.AsciiSerialize(*buf);
  
  shared_ptr<Blob> newBlob = Blob::CreateFromStream(*buf, 3);

  ASSERT_EQ(curBlob, *newBlob);
}

TEST_F(SerializeTest, SerializationString) {
  blob->AsciiSerialize(*buf);

  ASSERT_EQ(buf->str(), "3,2;5,3;\n");
}

TEST_F(SerializeTest, EmptySerializationString) {
  Blob curBlob(4, Blob::EIGHT_CONNECTED);
  curBlob.AsciiSerialize(*buf);

  ASSERT_EQ(buf->str(), "");
}

TEST_F(SerializeTest, CreateFromNormalString) {
  *buf << "5,3;3,2;\n";

  shared_ptr<Blob> newBlob = Blob::CreateFromStream(*buf, 3);
  ASSERT_EQ(newBlob->area(), 2);
  EXPECT_TRUE(newBlob->Contains(Point2i(5,3)));
  EXPECT_TRUE(newBlob->Contains(Point2i(3,2)));
  EXPECT_EQ(newBlob->minX(), 3);
  EXPECT_EQ(newBlob->maxX(), 5);
  EXPECT_EQ(newBlob->minY(), 2);
  EXPECT_EQ(newBlob->maxY(), 3);
}

TEST_F(SerializeTest, CreateFromEmptyString) {
  shared_ptr<Blob> newBlob = Blob::CreateFromStream(*buf, 3);

  ASSERT_EQ(newBlob->area(), 0);
}

TEST_F(SerializeTest, CreateFromStringWithMissingSemiColon) {
  *buf << "5,3;3,2\n";

  ASSERT_THROW(Blob::CreateFromStream(*buf, 3), ios_base::failure);
}

TEST_F(SerializeTest, CreateFromStringWithMissingNewline) {
  *buf << "5,3;3,2;";

  shared_ptr<Blob> newBlob = Blob::CreateFromStream(*buf, 3);
  ASSERT_EQ(newBlob->area(), 2);
  EXPECT_TRUE(newBlob->Contains(Point2i(5,3)));
  EXPECT_TRUE(newBlob->Contains(Point2i(3,2)));
  EXPECT_EQ(newBlob->minX(), 3);
  EXPECT_EQ(newBlob->maxX(), 5);
  EXPECT_EQ(newBlob->minY(), 2);
  EXPECT_EQ(newBlob->maxY(), 3);
}

TEST_F(SerializeTest, CreateFromStringWithExtraWhitespace) {
  *buf << "   5,3;3,2;   \n";

  ASSERT_THROW(Blob::CreateFromStream(*buf, 3), ios_base::failure);
}

TEST_F(SerializeTest, CreateFromBadStrings) {
  buf->str("5,3;a,b;\n");
  EXPECT_THROW(Blob::CreateFromStream(*buf, 3), ios_base::failure);
  buf->clear();

  buf->str("5,3;3,2;4,\n");
  EXPECT_THROW(Blob::CreateFromStream(*buf, 3), ios_base::failure);
  buf->clear();

  buf->str("5,3;3,2;,\n");
  EXPECT_THROW(Blob::CreateFromStream(*buf, 3), ios_base::failure);
  buf->clear();

  buf->str("monkey");
  EXPECT_THROW(Blob::CreateFromStream(*buf, 3), ios_base::failure);
  buf->clear();

  buf->str("dishwasher4,;");
  EXPECT_THROW(Blob::CreateFromStream(*buf, 3), ios_base::failure);
  buf->clear();
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
