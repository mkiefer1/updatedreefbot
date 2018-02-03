#include "visual_utility/TransformEstimator-Inl.h"

#include <gtest/gtest.h>
#include <boost/scoped_ptr.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cv_utils/DisplayImages.h"

using namespace cv;
using namespace std;
using namespace boost;

namespace visual_utility {

class EstimateAffineTransformTest : public ::testing::Test {

 protected:
  Mat_<double> image;
  scoped_ptr<AffineTransformEstimator> estimator;
  
  virtual void SetUp() {
    estimator.reset(new AffineTransformEstimator(100, 1e-6, 1.0));

    // Create a simple image
    image = Mat::zeros(20, 30, CV_64F);
    
    for (int i = 3; i < 14; i++) {
      image.at<double>(Point((i+10)/2, i)) += i/20.0;
    }

    for (int i = 9; i < 18; i++) {
      image.at<double>(Point(i, 6)) = 0.6;
    }
  }

  void ExpectMatrixNear(const Mat& a, const Mat& b, double error=1e-5) {
    ASSERT_EQ(a.rows, b.rows);
    ASSERT_EQ(a.cols, b.cols);

    for (int i = 0; i < a.rows; i++) {
      for (int j = 0; j < b.cols; j++) {
        EXPECT_NEAR(a.at<double>(Point(j,i)), b.at<double>(Point(j,i)), error);
      }
    }
  }

  Mat CreateWarpedImage(const Mat_<double>& src, const Mat& M) {
    return estimator->ApplyTransform(src, M, geo::BORDER_CONSTANT, 0.0);
  }

};

TEST_F(EstimateAffineTransformTest, SmallRotation) {
  double angle = 1.0;
  Mat desiredTransform = Mat::zeros(2,3,CV_64F);
  desiredTransform.at<double>(Point(0,0)) = cos(angle * M_PI / 180.0);
  desiredTransform.at<double>(Point(0,1)) = -sin(angle * M_PI /180.0);
  desiredTransform.at<double>(Point(1,0)) = sin(angle * M_PI / 180.0);
  desiredTransform.at<double>(Point(1,1)) = cos(angle * M_PI / 180.0);

  Mat warpedImage = CreateWarpedImage(image, desiredTransform);

  Mat foundTransform = estimator->EstimateTransform(image, warpedImage);         
  ExpectMatrixNear(desiredTransform, foundTransform);
}

TEST_F(EstimateAffineTransformTest, SmallTranslation) {
  Mat_<double> desiredTransform = Mat::eye(2,3,CV_64F);
  desiredTransform(0,2) = 2;
  desiredTransform(1,2) = -2;

  Mat warpedImage = CreateWarpedImage(image, desiredTransform);

  Mat foundTransform = estimator->EstimateTransform(image, warpedImage);         
  ExpectMatrixNear(desiredTransform, foundTransform);
}

TEST_F(EstimateAffineTransformTest, TranslationWithRescaling) {
  Mat_<double> desiredTransform = Mat::eye(2,3,CV_64F);
  desiredTransform(0,2) = 2;
  desiredTransform(1,2) = -2;

  Mat warpedImage = CreateWarpedImage(image, desiredTransform);

  estimator.reset(new AffineTransformEstimator(100, 1e-6, 2.0));
  Mat foundTransform = estimator->EstimateTransform(image, warpedImage);         
  ExpectMatrixNear(desiredTransform, foundTransform);
}

TEST_F(EstimateAffineTransformTest, ImageWithRescaling) {
  // Do a small rotation on an image that is resized
  double angle = 8.0;
  Mat_<double> desiredTransform = Mat::zeros(2,3,CV_64F);
  desiredTransform(0,0) = cos(angle * M_PI / 180.0);
  desiredTransform(0,1) = -sin(angle * M_PI /180.0);
  desiredTransform(1,0) = sin(angle * M_PI / 180.0);
  desiredTransform(1,1) = cos(angle * M_PI / 180.0);
  
  Mat biggerImage;
  resize(image, biggerImage, Size(), 2.0, 2.0, INTER_CUBIC);

  Mat warpedImage = CreateWarpedImage(biggerImage, desiredTransform);

  Mat foundTransform = estimator->EstimateTransform(biggerImage, warpedImage);         
  ExpectMatrixNear(desiredTransform, foundTransform);
}


TEST_F(EstimateAffineTransformTest, TestIdentityTransform) {
  Mat desiredTransform = Mat::eye(2,3,CV_64F);

  Mat warpedImage = CreateWarpedImage(image, desiredTransform);

  Mat foundTransform = estimator->EstimateTransform(image, warpedImage);
                                                
  ExpectMatrixNear(desiredTransform, foundTransform);
  ExpectMatrixNear(warpedImage, image);
}

TEST_F(EstimateAffineTransformTest, TestNoTransform) {

  Mat warpedImage = image.clone();

  Mat foundTransform = estimator->EstimateTransform(image, warpedImage);

  ExpectMatrixNear(Mat::eye(2,3,CV_64F), foundTransform);
}

// mdesnoyer: disable the test because OpenCV isn't throwing a normal
// exception, it's killing the program.
TEST_F(EstimateAffineTransformTest, DISABLED_SrcNotGreyscale) {
  Mat colorImage;

  cvtColor(image, colorImage, CV_GRAY2BGR);

  EXPECT_THROW(estimator->EstimateTransform(colorImage, image), cv::Exception);
}

// mdesnoyer: disable the test because OpenCV isn't throwing a normal
// exception, it's killing the program.
TEST_F(EstimateAffineTransformTest, DISABLED_DestNotGreyscale) {
  Mat colorImage;

  cvtColor(image, colorImage, CV_GRAY2BGR);

  EXPECT_THROW(estimator->EstimateTransform(image, colorImage), cv::Exception);
}

TEST_F(EstimateAffineTransformTest, ReadDataTest) {
  Mat img1 = imread("test/testFishFrame1.png", -1);
  Mat img2 = imread("test/testFishFrame2.png", -1);

  Mat img1Float;
  Mat img2Float;
  img1.convertTo(img1Float, CV_64F, 1./255);
  img2.convertTo(img2Float, CV_64F, 1./255);

  estimator.reset(new AffineTransformEstimator(50, 1e-8, 4.0));

  Mat M = estimator->EstimateTransform(img1Float, img2Float);
  ASSERT_EQ(M.rows, 2);
  ASSERT_EQ(M.cols, 3);
}

TEST_F(EstimateAffineTransformTest, RealImageRotateAndTranslate) {
  // Rotate
  double angle = 0.5;
  Mat_<double> desiredTransform = Mat::zeros(2,3,CV_64F);
  desiredTransform(0,0) = cos(angle * M_PI / 180.0);
  desiredTransform(0,1) = -sin(angle * M_PI /180.0);
  desiredTransform(1,0) = sin(angle * M_PI / 180.0);
  desiredTransform(1,1) = cos(angle * M_PI / 180.0);

  // Zoom
  desiredTransform(0,0) *= 1.02;
  desiredTransform(1,1) *= 1.05;
  
  // Shift
  desiredTransform(0,2) = 1;
  desiredTransform(1,2) = -2;

  estimator.reset(new AffineTransformEstimator(100, 1e-8, 4.0));

  Mat img1 = imread("test/hima_set22_404.bmp", 0);
  Mat img1Float;
  img1.convertTo(img1Float, CV_64F, 1./255);

  Mat warpedImage = CreateWarpedImage(img1Float, desiredTransform);

  Mat foundTransform = estimator->EstimateTransform(img1Float, warpedImage);

  ExpectMatrixNear(desiredTransform, foundTransform);

  cv_utils::DisplayNormalizedImage(img1, "img1");
  cv_utils::DisplayNormalizedImage(warpedImage, "img2");
  cv_utils::DisplayNormalizedImage(abs(warpedImage - CreateWarpedImage(img1Float,
                                                                       foundTransform)),
                                   "diff");
  cv_utils::DisplayNormalizedImage(CreateWarpedImage(img1Float,
                                                     foundTransform),
                                   "warped");
  cv_utils::ShowWindowsUntilKeyPress();
}

}; // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
