#include "objectness/objectness.h"

// Matlab includes
#include "engine.h"
#include "matrix.h"

using namespace std;
using namespace cv;

namespace objectness {

// A singleton wrapper class to handle the construction and destruction
// properly
class MatlabEngine {
public:
  MatlabEngine()
    : engine_(NULL) {}
  ~MatlabEngine() {
    CloseEngine();
  }

  Engine* get() {
    if (engine_ == NULL) {
      ROS_INFO_STREAM("Initializing the MATLAB engine");
      engine_ = engOpen(NULL);
      if (engine_ == NULL) {
        ROS_FATAL_STREAM("Could not start the MATLAB engine");
      }
      ROS_INFO_STREAM("MATLAB engine running");
    }
    return engine_;
  }

  void CloseEngine() {
    if (engine_ != NULL) {
      ROS_INFO_STREAM("Closing the matlab engine");
      engClose(engine_);
      engine_ = NULL;
    }
  }

private:
  Engine* engine_;
};
MatlabEngine _matlabEngine_;

Objectness::~Objectness() {
}

bool Objectness::Init() {
  return InitMatlab();
}

bool Objectness::InitMatlab() {
  ROS_INFO_STREAM("Setting up the MATLAB environment");
  engOutputBuffer(_matlabEngine_.get(), matlabBuffer_, 1023);
  matlabBuffer_[1023] = 0;
  string curDir = FindPackageDir();
  if (engEvalString(_matlabEngine_.get(), ("addpath(genpath('" + curDir +
                                    "/src'));global felzen_dir;felzen_dir = '"
                                    + curDir +
                                    "/src/ObjectnessICCV/segmentation/';").c_str())) {
    ROS_FATAL_STREAM("Error setting up the MATLAB environement: "
                     << matlabBuffer_);
    return false;
  }

  ROS_INFO_STREAM("Matlab successfully started");
  return true;
}

string Objectness::FindPackageDir() const {
  string result = "";

  FILE* pipe = popen("rospack find objectness", "r");
  if (!pipe) {
    ROS_FATAL_STREAM("Could not execute rospack to find where this package is "
                     "installed");
    return "ERROR";
  }
  char buffer[128];
  while (!feof(pipe)) {
    if (fgets(buffer, 128, pipe) != NULL) {
      result += buffer;
    }
  }
  pclose(pipe);
  result.erase(result.size()-1);
  return result;
}

void Objectness::CalculateObjectness(const cv::Mat& image,
                                     const std::vector<cv::Rect>& rois,
                                     std::vector<ROIScore>* scoreOut,
                                     double *runtime) {
  ROS_ASSERT(scoreOut);

  if (image.channels() != 3 || image.depth() != CV_8U) {
    ROS_ERROR("Image must be BGR and 8-bit, but it was not");
    return;
  }

  // Create the matlab image object
  mwSize dims[3];
  dims[0] = image.rows;
  dims[1] = image.cols;
  dims[2] = 3;
  mxArray* matlabImage = mxCreateNumericArray(3, dims, mxUINT8_CLASS, mxREAL);
  uint8_t* matImagePtr = (uint8_t*)mxGetData(matlabImage);
  for (int row = 0; row < image.rows; ++row) {
    const Vec3b* cvPtr = image.ptr<Vec3b>(row);
    for (int col = 0; col < image.cols; ++col) {
      matImagePtr[col*dims[0] + row] = (*cvPtr)[2];
      matImagePtr[dims[0] * (dims[1] + col) + row] = (*cvPtr)[1];
      matImagePtr[dims[0] * (2 * dims[1] + col) + row] = (*cvPtr)[0];
    
      cvPtr++;
    }
  }

  // Create the set of windows to evaluate
  mxArray* matWindows = mxCreateNumericMatrix(rois.size(), 4, mxDOUBLE_CLASS,
                                              mxREAL);
  double* matWinPtr = (double*)mxGetData(matWindows);
  unsigned int nBoxes = rois.size();
  for (unsigned int row = 0u; row < nBoxes; ++row) {
    const Rect& roi = rois[row];
    matWinPtr[row] = roi.x + 1;
    matWinPtr[nBoxes + row] = roi.y + 1;
    matWinPtr[2*nBoxes + row] = roi.x + roi.width;
    matWinPtr[3*nBoxes + row] = roi.y + roi.height;
  }

  // Send the image and regions to the matlab engine
  engPutVariable(_matlabEngine_.get(), "img", matlabImage);
  engPutVariable(_matlabEngine_.get(), "initialWindows", matWindows);
  mxDestroyArray(matlabImage);
  mxDestroyArray(matWindows);

  engEvalString(_matlabEngine_.get(), "save('test.mat', 'img', 'initialWindows')");

  // Start the timer
  ros::Time startTime;
  if (doTiming_) {
    startTime = ros::Time::now();
  }

  // Calculate the scores for all the windows
  if (engEvalString(_matlabEngine_.get(),
                    "scores = mvg_scoreWindows(img, initialWindows);")) {
    ROS_ERROR_STREAM("Error scoring the image:"
                     << matlabBuffer_
                     << " Restarting MATLAB");
    _matlabEngine_.CloseEngine();
    InitMatlab();
    return;
  }

  if (doTiming_ && runtime) {
    *runtime = (ros::Time::now() - startTime).toSec();
  }

  // Get the scores back from matlab
  mxArray* scoreArray = engGetVariable(_matlabEngine_.get(), "scores");
  if (scoreArray) {
    unsigned int nScores = mxGetM(scoreArray);
    if (nScores != nBoxes) {
      ROS_ERROR_STREAM("The number of scores returned is not the same as "
                       "the number of windows. AHHHH");
      return ;
    }

    double* scorePtr = (double*)mxGetData(scoreArray);
    for (unsigned int i = 0u; i < nBoxes; ++i) {
      scoreOut->push_back(ROIScore(scorePtr[i], rois[i]));
    }
  } else {
    ROS_ERROR_STREAM("Could not find the score variable. "
                     << "Odds are that there was a matlab error: "
                     << matlabBuffer_);
  }
}

} // objectness
