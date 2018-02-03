#include "visual_utility/cvutils.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <boost/scoped_ptr.hpp>

#include "visual_utility/gnuplot_i.hpp"


using namespace cv;

namespace cvutils {

boost::scoped_ptr<Gnuplot> plotter_;

Gnuplot& plotter() {
  if (plotter_.get() == NULL) {
    plotter_.reset(new Gnuplot());
  }
  return *plotter_;
}

// Writes an image to a file but first normalizes it to the 0-255 range
void WriteNormalizedImage(const string filename, const Mat& image) {
  Mat outImage;
  normalize(image, outImage, 0, 255, NORM_MINMAX,
            CV_MAKETYPE(CV_8U, image.channels()));
  imwrite(filename, outImage);
}

void DisplayNormalizedImage(const cv::Mat& image, const char* windowName) {
  Mat outImage = image;
  if (image.channels() == 1) {
    normalize(image, outImage, 0, 255, NORM_MINMAX,
              CV_MAKETYPE(CV_8U, image.channels()));
  }

  
  cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);
  
  cvShowImage(windowName, &IplImage(outImage));
}

void DisplayLabImage(const cv::Mat& image, const char* windowName) {
  Mat outImage;
  Mat outImage32;
  image.convertTo(outImage32, CV_32FC3);
  cvtColor(outImage32, outImage, CV_Lab2BGR);
  
  cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);
  
  cvShowImage(windowName, &IplImage(outImage));
}

void ShowWindowsUntilKeyPress() {
  int key = -1;
  while (key < 0) {
    key = cvWaitKey(100);
  }
}

}
