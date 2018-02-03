#include "visual_utility/VisualUtilityEstimator.h"
#include "visual_utility/VisualUtilityMosaic.h"
#include "visual_utility/FrameEstimator.h"
#include "visual_utility/cvutils.h"

#include <gtest/gtest.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <math.h>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace cvutils;

namespace visual_utility {

TEST(ManualInspection, DISABLED_UndergroundImage) {
  SpectralSaliency estimator;

  Mat testImage = imread("test/testUnderground1.bmp");

  DisplayNormalizedImage(testImage, "original");
  DisplayNormalizedImage(estimator.CalculateVisualUtility(testImage, 0.0),
                         "saliency");
  ShowWindowsUntilKeyPress();
}

TEST(ManualInpsect, DISABLED_WithHighEntropyFraming) {
  SpectralSaliency estimator;
  NullVUMosaic mosaic(8, 15);
  HighRelativeEntropy frameEstimator(1.0, 200, 0.05);
  frameEstimator.DisplayDebugImages(true);

  Mat testImage = imread("test/testUnderground1.bmp");
  DisplayNormalizedImage(testImage, "original");

  Mat_<double> visualUtility = estimator.CalculateVisualUtility(testImage, 0.0);
  DisplayNormalizedImage(visualUtility, "saliency");

  mosaic.AddFrame(visualUtility, NULL);
  DisplayNormalizedImage(mosaic.ExtractRegion(Rect(-testImage.cols/2,
                                                   -testImage.rows/2,
                                                   testImage.cols,
                                                   testImage.rows)),
                         "Smoothed mosaic");

  vector<FrameOfInterest> foi;
  frameEstimator.FindInterestingFrames(mosaic, &foi);
  
  ShowWindowsUntilKeyPress();
    
}

bool sortDesc (double i, double j) { return i>j; }

// Shows the top 10 boxes in a frame
void ShowGridBoxes(const Mat& image, VisualUtilityEstimator& vuEstimator,
                   const string& winName) {

  Mat overlayFrame = image.clone();

  Mat gridResults = vuEstimator.CalculateVisualUtility(image,
                                                       0, 0,
                                                       32, 32,
                                                       16, 16,
                                                       1.5, 1.5,
                                                       false,
                                                       2.0/30);

  // Sort the scores of the boxes using STL hackery because the opencv
  // sortIdx function doesn't seem to do anything.
  vector<double> flatResults(gridResults.begin<double>(),
                             gridResults.end<double>());
  vector<double> sortedFlatResults(flatResults);
  partial_sort(sortedFlatResults.begin(), sortedFlatResults.begin()+10,
               sortedFlatResults.end(), sortDesc);
  
  // Draw rectangles for the top 10 boxes
  for (vector<double>::iterator i=flatResults.begin();
       i < flatResults.end(); ++i) {
    i = find_if(i, flatResults.end(),
                bind2nd(greater_equal<double>(), sortedFlatResults[9]));

    int sortedIdx = i - flatResults.begin();
    
    int wI = sortedIdx * gridResults.elemSize() / gridResults.step[0];
    int hI = (sortedIdx * gridResults.elemSize() - wI*gridResults.step[0]) /
      gridResults.step[1];
    int xI = (sortedIdx * gridResults.elemSize()  - wI*gridResults.step[0] 
              - hI*gridResults.step[1]) /
      gridResults.step[2];
    int yI = sortedIdx % gridResults.size[3];

    Rect rect(16*xI, 16*yI, cvRound(32*pow(1.5,wI)),\
              cvRound(32*pow(1.5,hI)));
    rectangle(overlayFrame, rect, Scalar(0.0, 0.0, 255, 1.0));
  }

  DisplayNormalizedImage(overlayFrame, winName.c_str());
  
}

void ShowVisualUtilities(string f1, string f2) {
  AffineTransformEstimator transformEstimator(100,
                                              1e-6,
                                              4.0);

  RelativeEntropyVUWrapper salEstimator(new SpectralSaliency());
  RelativeEntropyVUWrapper motionEstimator(
    new LABMotionVUEstimator(transformEstimator, 0.05, 20.0, 3));
  NullVUMosaic motionMosaic(8, 15);
  NullVUMosaic salMosaic(8, 15);

  Mat im1 = imread(f1);
  Mat im2 = imread(f2);
  DisplayNormalizedImage(im2, "original");
  DisplayNormalizedImage(im1, "im1");
  
  Mat_<double> saliency = salEstimator.CalculateVisualUtility(im2, 0.0);
  salMosaic.AddFrame(saliency, NULL);
  Mat_<double> logSaliency;
  log(salMosaic.ExtractRegion(Rect(-im2.cols/2, -im2.rows/2,
                                   im2.cols, im2.rows)), logSaliency);
  DisplayNormalizedImage(logSaliency,
                         "Saliency");
  ShowGridBoxes(im2, salEstimator, "Saliency Boxes");

  motionEstimator.CalculateVisualUtility(im1, 0.0);
  Mat_<double> motion = motionEstimator.CalculateVisualUtility(im2, 1.0/30);
  DisplayNormalizedImage(motion, "raw motion");
  motionMosaic.AddFrame(motion, NULL);
  Mat_<double> logMotion;
  log(motionMosaic.ExtractRegion(Rect(-im2.cols/2, -im2.rows/2,
                                      im2.cols, im2.rows)), logMotion);
  DisplayNormalizedImage(motionMosaic.ExtractRegion(Rect(-im2.cols/2, -im2.rows/2,
                                                         im2.cols,
                                                         im2.rows)),
                         "Motion");
  ShowGridBoxes(im1, motionEstimator, "Motion Boxes");

  ShowWindowsUntilKeyPress();
}

TEST(MakeGraphics, fish) {
  ShowVisualUtilities("test/fish_frame1.jpg", "test/fish_frame2.jpg");
}

TEST(MakeGraphics, fish_grey) {
  ShowVisualUtilities("test/testFishFrame1.png", "test/testFishFrame2.png");
}

TEST(MakeGraphics, eth) {
  ShowVisualUtilities("test/eth_set02_110.png", "test/eth_set02_111.png");
}

TEST(MakeGraphics, set22) {
  //ShowVisualUtilities("test/hima_set22_404.bmp", "test/hima_set22_405.bmp");
  ShowVisualUtilities("test/hima_set22_010.bmp", "test/hima_set22_011.bmp");
}

TEST(MakeGraphics, caltech) {
  ShowVisualUtilities("test/caltech_set06_V015_1529.jpg", "test/caltech_set06_V015_1533.jpg");
}

TEST(MakeGraphics, set6) {
  ShowVisualUtilities("test/hima_set6_010.bmp", "test/hima_set6_011.bmp");
}


} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ros::Time::init();
  return RUN_ALL_TESTS();
}
