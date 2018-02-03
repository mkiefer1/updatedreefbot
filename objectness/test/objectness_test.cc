#include "objectness/objectness.h"
#include <gtest/gtest.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>
#include <math.h>

#include "cv_utils/DisplayImages.h"

using namespace cv;

namespace objectness {

void GenerateWindowSampling(const Mat& image, double minWidth,
                            double minHeight, int winStride,
                            double scaleStride,
                            vector<Rect>* out) {
  ASSERT_TRUE(out);

  double curWidth = minWidth;
  double curHeight = minHeight;
  while (curWidth < image.cols && curHeight < image.rows) {
    for (int x = 0; x < image.cols - curWidth; x += winStride) {
      for (int y = 0; y < image.rows - curHeight; y += winStride) {
        out->push_back(Rect(round(x), round(y), round(curWidth),
                            round(curHeight)));
      }
    }
    
    curWidth *= scaleStride;
    curHeight *= scaleStride;
  }
}

void DisplaySelectedWindows(const string& imgFile, double thresh) {

  // Open the image
  Mat image = imread(imgFile);

  vector<Rect> windows;
  GenerateWindowSampling(image, 64, 128, 10, 1.10, &windows);

  Objectness calculator(false);
  vector<Objectness::ROIScore> scores;
  ASSERT_TRUE(calculator.Init());
  calculator.CalculateObjectness(image, windows, &scores, NULL);

  // Draw the rectangles on the image
  for (vector<Objectness::ROIScore>::const_iterator i = scores.begin();
       i != scores.end(); ++i) {
    if (i->first > thresh) {
      cv::rectangle(image, Point(i->second.x, i->second.y),
                    Point(i->second.x + i->second.width,
                          i->second.y + i->second.height),
                    CV_RGB(255, 0, 0),
                    2); // thickness
    }
  }

  // Show the image
  cv_utils::DisplayNormalizedImage(image);
  cv_utils::ShowWindowsUntilKeyPress();

}

TEST(DisplayImageTest, Eth) {
  DisplaySelectedWindows("test/eth_set02_111.png", 0.9);
}

TEST(DisplayImageTest, Steetcar) {
  DisplaySelectedWindows("src/ObjectnessICCV/exampleImage.jpg", 0.8);
}

} // namespace 

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
