#include "cv_blobs/BlobResult.h"
#include "cv_blobs/BlobResult-Inline.h"
#include <boost/scoped_ptr.hpp>
#include <gtest/gtest.h>
#include "opencv2/core/core.hpp"
#include <sstream>

#include "cv_blobs/BlobFilters.h"

using namespace cv;
using namespace boost;

namespace cv_blobs {

class FloatImageBlobTest : public ::testing::Test {
 protected:
  Mat image;

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  virtual void SetUp() {
    // Create a 10 by 10 image of all zeros
    image = Mat::zeros(10, 10, CV_32FC1);
  }

};

TEST_F(FloatImageBlobTest, NoBlobs) {
  BlobResult<float> result(image, 100);
  ASSERT_EQ(result.nBlobs(), 0);
}

TEST_F(FloatImageBlobTest, OnePositiveBlob) {
  for (int row = 2; row < 4; row++) {
    for (int col = 3; col < 5; ++col) {
      image.at<float>(row, col) = 101;
    }
  }
  BlobResult<float> result(image, 100);
  ASSERT_EQ(result.nBlobs(), 1);
  const Blob& blob = result.GetBlob(0);
  ASSERT_EQ(blob.area(), 4);
  EXPECT_EQ(blob.minX(), 3);
  EXPECT_EQ(blob.maxX(), 4);
  EXPECT_EQ(blob.minY(), 2);
  EXPECT_EQ(blob.maxY(), 3);
}

TEST_F(FloatImageBlobTest, OneNegativeBlob) {
  for (int row = 2; row < 4; row++) {
    for (int col = 3; col < 5; ++col) {
      image.at<float>(row, col) = -50;
    }
  }
  BlobResult<float> result(image, -1, Blob::EIGHT_CONNECTED, false);
  ASSERT_EQ(result.nBlobs(), 1);
  const Blob& blob = result.GetBlob(0);
  ASSERT_EQ(blob.area(), 4);
  EXPECT_EQ(blob.minX(), 3);
  EXPECT_EQ(blob.maxX(), 4);
  EXPECT_EQ(blob.minY(), 2);
  EXPECT_EQ(blob.maxY(), 3);
}

TEST_F(FloatImageBlobTest, One2PointAdjBlob) {
  image.at<float>(3,4) = 101;
  image.at<float>(4,4) = 101;

  BlobResult<float> result(image, 100, Blob::FOUR_CONNECTED);
  ASSERT_EQ(result.nBlobs(), 1);
  const Blob& blob = result.GetBlob(0);
  ASSERT_EQ(blob.area(), 2);
  EXPECT_EQ(blob.minX(), 4);
  EXPECT_EQ(blob.maxX(), 4);
  EXPECT_EQ(blob.minY(), 3);
  EXPECT_EQ(blob.maxY(), 4);
}

TEST_F(FloatImageBlobTest, One2PointDiagBlob) {
  image.at<float>(3,4) = 101;
  image.at<float>(4,5) = 101;

  BlobResult<float> eightResult(image, 100, Blob::EIGHT_CONNECTED);
  ASSERT_EQ(eightResult.nBlobs(), 1);
  const Blob& eightBlob = eightResult.GetBlob(0);
  ASSERT_EQ(eightBlob.area(), 2);
  EXPECT_EQ(eightBlob.minX(), 4);
  EXPECT_EQ(eightBlob.maxX(), 5);
  EXPECT_EQ(eightBlob.minY(), 3);
  EXPECT_EQ(eightBlob.maxY(), 4);

  BlobResult<float> fourResult(image, 100, Blob::FOUR_CONNECTED);
  ASSERT_EQ(fourResult.nBlobs(), 2);
  EXPECT_EQ(fourResult.GetBlob(0).area(), 1);
  EXPECT_EQ(fourResult.GetBlob(1).area(), 1);
}

TEST_F(FloatImageBlobTest, OneBlobEightConnected) {
  for (int row = 2; row < 4; row++) {
    for (int col = row-1; col < 5; col +=2) {
      image.at<float>(row, col) = 101;
    }
  }
  BlobResult<float> result(image, 100, Blob::EIGHT_CONNECTED);
  ASSERT_EQ(result.nBlobs(), 1);
  const Blob& blob = result.GetBlob(0);
  ASSERT_EQ(blob.area(), 4);
  EXPECT_EQ(blob.minX(), 1);
  EXPECT_EQ(blob.maxX(), 4);
  EXPECT_EQ(blob.minY(), 2);
  EXPECT_EQ(blob.maxY(), 3);
}

TEST_F(FloatImageBlobTest, SearchFourConnectedWhenActuallyEight) {
  for (int row = 2; row < 4; row++) {
    for (int col = row-1; col < 5; col +=2) {
      image.at<float>(row, col) = 101;
    }
  }
  BlobResult<float> result(image, 100, Blob::FOUR_CONNECTED);
  ASSERT_EQ(result.nBlobs(), 4);
  for (int i = 0; i < 4; ++i) {
    const Blob& blob = result.GetBlob(i);
    EXPECT_EQ(blob.area(), 1);
  }
}

TEST_F(FloatImageBlobTest, FilterOutLargeBlobs) {
  // Creates a square 2x2 blob
  for (int row = 2; row < 4; row++) {
    for (int col = 3; col < 5; ++col) {
      image.at<float>(row, col) = 101;
    }
  }

  // Create a single pixel blob
  image.at<float>(8,9) = 105;

  BlobResult<float> result(image, 100);
  ASSERT_EQ(result.nBlobs(), 2);
  result.Filter(AreaCompare<less<int>, int>(2));
  ASSERT_EQ(result.nBlobs(), 1);
  const Blob& blob = result.GetBlob(0);
  ASSERT_EQ(blob.area(), 1);
  EXPECT_EQ(blob.minX(), 9);
  EXPECT_EQ(blob.maxX(), 9);
  EXPECT_EQ(blob.minY(), 8);
  EXPECT_EQ(blob.maxY(), 8);
}

TEST_F(FloatImageBlobTest, FilterOutSmallBlobs) {
  // Creates a square 2x2 blob
  for (int row = 2; row < 4; row++) {
    for (int col = 3; col < 5; ++col) {
      image.at<float>(row, col) = 101;
    }
  }

  // Create a single pixel blob
  image.at<float>(8,9) = 105;

  BlobResult<float> result(image, 100);
  ASSERT_EQ(result.nBlobs(), 2);
  result.Filter(AreaCompare<greater<int>, int>(2));
  ASSERT_EQ(result.nBlobs(), 1);
  const Blob& blob = result.GetBlob(0);
  ASSERT_EQ(blob.area(), 4);
  EXPECT_EQ(blob.minX(), 3);
  EXPECT_EQ(blob.maxX(), 4);
  EXPECT_EQ(blob.minY(), 2);
  EXPECT_EQ(blob.maxY(), 3);
}

TEST_F(FloatImageBlobTest, ToImageTest) {
  // Creates a square 2x2 blob
  for (int row = 2; row < 4; row++) {
    for (int col = 3; col < 5; ++col) {
      image.at<float>(row, col) = 101;
    }
  }

  // Create a single pixel blob
  image.at<float>(8,9) = 105;

  BlobResult<float> result(image, 100);
  Mat binaryImage = result.ToImage();
  ASSERT_EQ(binaryImage.channels(), 1);
  ASSERT_EQ(binaryImage.depth(), CV_8U);

  EXPECT_GE(binaryImage.at<uchar>(2,3), 1);
  EXPECT_GE(binaryImage.at<uchar>(3,3), 1);
  EXPECT_GE(binaryImage.at<uchar>(2,4), 1);
  EXPECT_GE(binaryImage.at<uchar>(3,4), 1);
  EXPECT_GE(binaryImage.at<uchar>(8,9), 1);
  EXPECT_EQ(binaryImage.at<uchar>(2,2), 0);
  EXPECT_EQ(binaryImage.at<uchar>(2,5), 0);
  EXPECT_EQ(binaryImage.at<uchar>(1,3), 0);
  EXPECT_EQ(binaryImage.at<uchar>(3,2), 0);
  EXPECT_EQ(binaryImage.at<uchar>(9,8), 0);
  EXPECT_EQ(binaryImage.at<uchar>(8.8), 0);
}

TEST_F(FloatImageBlobTest, ToBinaryImageTest) {
  // Creates a square 2x2 blob
  for (int row = 2; row < 4; row++) {
    for (int col = 3; col < 5; ++col) {
      image.at<float>(row, col) = 101;
    }
  }

  // Create a single pixel blob
  image.at<float>(8,9) = 105;

  BlobResult<float> result(image, 100);
  Mat binaryImage = result.ToBinaryImage();
  ASSERT_EQ(binaryImage.channels(), 1);
  ASSERT_EQ(binaryImage.depth(), CV_8U);

  EXPECT_EQ(binaryImage.at<uchar>(2,3), 255);
  EXPECT_EQ(binaryImage.at<uchar>(3,3), 255);
  EXPECT_EQ(binaryImage.at<uchar>(2,4), 255);
  EXPECT_EQ(binaryImage.at<uchar>(3,4), 255);
  EXPECT_EQ(binaryImage.at<uchar>(8,9), 255);
  EXPECT_EQ(binaryImage.at<uchar>(2,2), 0);
  EXPECT_EQ(binaryImage.at<uchar>(2,5), 0);
  EXPECT_EQ(binaryImage.at<uchar>(1,3), 0);
  EXPECT_EQ(binaryImage.at<uchar>(3,2), 0);
  EXPECT_EQ(binaryImage.at<uchar>(9,8), 0);
  EXPECT_EQ(binaryImage.at<uchar>(8.8), 0);
}

class SerializeTest : public FloatImageBlobTest {
protected:
  scoped_ptr<BlobResult<float> > result;
  BlobResultSerializer<float> serializer;
  scoped_ptr<stringstream> buf;

  virtual void SetUp() {
    FloatImageBlobTest::SetUp();
    buf.reset(new stringstream());

    // Add a couple of blobs to the image
    image.at<float>(3,4) = 105;
    image.at<float>(4,4) = 105;
    image.at<float>(8,9) = 105;

    // Find the blobs
    result.reset(new BlobResult<float>(image, 100));
    ASSERT_EQ(result->nBlobs(), 2);
  }

  // Checks that the given string deserializes to the default BlobResult
  void TestStringDeserialization(const string& str, const string& expectedImgName) {
    SCOPED_TRACE(str);
    string imgName;

    buf.reset(new stringstream());
    buf->str(str);

    shared_ptr<BlobResult<float> > newResult = serializer.Deserialize(*buf,
                                                                      &imgName);

    EXPECT_EQ(expectedImgName, imgName);
    EXPECT_EQ(newResult->nBlobs(), result->nBlobs());
    for (int i = 0; i < result->nBlobs(); i++) {
      EXPECT_NE(&newResult->GetBlob(i), &result->GetBlob(i));
      EXPECT_EQ(newResult->GetBlob(i), result->GetBlob(i));
    }
  }
};

TEST_F(SerializeTest, RoundTripSerialize) {
  string imgName;

  serializer.Serialize(*buf, *result, string("test/testImage.jpg"));

  shared_ptr<BlobResult<float> > newResult = serializer.Deserialize(*buf,
                                                                    &imgName);

  EXPECT_EQ(imgName, "test/testImage.jpg");
  ASSERT_EQ(newResult->nBlobs(), result->nBlobs());
  for (int i = 0; i < result->nBlobs(); i++) {
    EXPECT_NE(&newResult->GetBlob(i), &result->GetBlob(i));
    EXPECT_EQ(newResult->GetBlob(i), result->GetBlob(i));
  }

}

TEST_F(SerializeTest, NormalSerializationString) {
  serializer.Serialize(*buf, *result, "test/testImage.jpg");
  
  // The order here shouldn't actually matter, but encoding that as a
  // test is problamatic. So this test will only make sure that the
  // string doesn't change.
  EXPECT_EQ(buf->str(), "test/testImage.jpg\n9,8;\n4,3;4,4;\n");
}

TEST_F(SerializeTest, NoBlobsString) {
  // Since the values are 105, they won't be added to the blob
  BlobResult<float> newResult(image, 300);

  serializer.Serialize(*buf, newResult, "image1");

  EXPECT_EQ(buf->str(), "image1\n");
}

TEST_F(SerializeTest, ValidDeserialization) {
  TestStringDeserialization("test/testImage.jpg\n9,8;\n4,3;4,4;\n",
                            "test/testImage.jpg");
  TestStringDeserialization("test/testImage.jpg\n9,8;\n4,4;4,3;\n",
                            "test/testImage.jpg");
}

TEST_F(SerializeTest, BlobOrderCorrect) {
  string imgName;

  buf->str("test/testImage.jpg\n4,4;4,3;\n9,8;\n");

  shared_ptr<BlobResult<float> > newResult = serializer.Deserialize(*buf,
                                                                    &imgName);
  EXPECT_EQ(imgName, "test/testImage.jpg");
  ASSERT_EQ(newResult->nBlobs(), result->nBlobs());
  EXPECT_NE(&newResult->GetBlob(0), &result->GetBlob(0));
  EXPECT_NE(newResult->GetBlob(0), result->GetBlob(0));
  EXPECT_EQ(newResult->GetBlob(0), result->GetBlob(1));
  EXPECT_NE(&newResult->GetBlob(1), &result->GetBlob(1));
  EXPECT_NE(newResult->GetBlob(1), result->GetBlob(1));
  EXPECT_EQ(newResult->GetBlob(1), result->GetBlob(0));
}

TEST_F(SerializeTest, NoImageName) {
  string imgName;
  buf->str("5,3;3,4;\n");
  EXPECT_THROW(serializer.Deserialize(*buf, &imgName), ios_base::failure);
}

TEST_F(SerializeTest, InvalidImageName) {
  string imgName;
  buf->str("my_unknown_image.jpg\n5,3;3,4;\n");
  EXPECT_THROW(serializer.Deserialize(*buf, &imgName), ios_base::failure);
}

TEST_F(SerializeTest, CreateFromBadStrings) {
  string imgName;

  buf->str("test/testImage.jpg\n5,3;a,b;\n");
  EXPECT_THROW(serializer.Deserialize(*buf, &imgName), ios_base::failure);
  buf->clear();

  buf->str("test/testImage.jpg\n5,3;3,2;\n7,3\n");
  EXPECT_THROW(serializer.Deserialize(*buf, &imgName), ios_base::failure);
  buf->clear();

  buf->str("test/testImage.jpg\n5,3;3,2;\n7\n");
  EXPECT_THROW(serializer.Deserialize(*buf, &imgName), ios_base::failure);
  buf->clear();

  buf->str("test/testImage.jpg\n5,3;3,2;\n7;\n");
  EXPECT_THROW(serializer.Deserialize(*buf, &imgName), ios_base::failure);
  buf->clear();

  buf->str("test/testImage.jpg\n5,3;3,2;\n7,3;");
  EXPECT_THROW(serializer.Deserialize(*buf, &imgName), ios_base::failure);
  buf->clear();

  buf->str("test/testImage.jpg\n5,3;3,2\n7,3;\n");
  EXPECT_THROW(serializer.Deserialize(*buf, &imgName), ios_base::failure);
  buf->clear();

  buf->str("test/testImage.jpg\n5,3;3,2;\n7,3");
  EXPECT_THROW(serializer.Deserialize(*buf, &imgName), ios_base::failure);
  buf->clear();
}

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
