#include "cascade_parts_detector/cascade_parts_detector.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <boost/lexical_cast.hpp>

#include <sensor_msgs/RegionOfInterest.h>
#include <std_msgs/Duration.h>

using namespace sensor_msgs;
using namespace ros;
using namespace std;
using namespace boost;

namespace cascade_parts_detector {

CascadePartsDetector::~CascadePartsDetector() {
  CloseMatlab();
}

void CascadePartsDetector::CloseMatlab() {
  if (matlabEngine_ != NULL) {
    ROS_INFO_STREAM("Closing the matlab engine");
    engClose(matlabEngine_);
    matlabEngine_ = NULL;
  }
}

bool CascadePartsDetector::Init(const std::string& modelFile, double thresh,
                                bool doCascade, bool doTiming) {
  modelFile_ = modelFile;
  thresh_ = lexical_cast<string>(thresh);
  doCascade_ = doCascade;
  doTiming_ = doTiming;

  if (!InitMatlab()) {
    return false;
  }
  if (!InitROS()) {
    return false;
  }
  return true;
}

bool CascadePartsDetector::InitMatlab() {
  ROS_INFO_STREAM("Initializing the MATLAB engine");
  matlabEngine_ = engOpen(NULL);
  if (matlabEngine_ == NULL) {
    ROS_FATAL_STREAM("Could not start the MATLAB engine");
    return false;
  }

  ROS_INFO_STREAM("Setting up the MATLAB environment");
  engOutputBuffer(matlabEngine_, matlabBuffer_, 1023);
  matlabBuffer_[1023] = 0;
  if (engEvalString(matlabEngine_, ("addpath(genpath('" + FindPackageDir() +
                                    "/src/voc_release4'))").c_str())) {
    ROS_FATAL_STREAM("Error setting up the MATLAB environement: "
                     << matlabBuffer_);
    return false;
  }
  

  ROS_INFO_STREAM("Loading the parts model " << modelFile_);
  if (engEvalString(matlabEngine_, (string("load ") + modelFile_).c_str())) {
    ROS_FATAL_STREAM("Error loading the parts model: " << matlabBuffer_);
    return false;
  }

  if (doCascade_) {
    ROS_INFO_STREAM("Converting model to a cascade model");
    if (engEvalString(matlabEngine_,
                      ("csc_model = cascade_model(model, '2009', 5, " +
                       thresh_ + ");").c_str())) {
      ROS_FATAL_STREAM("Could not convert model to a cascade model");
      return false;
    }
  }

  ROS_INFO_STREAM("Matlab successfully started and the parts model is loaded");
  return true;
}

bool CascadePartsDetector::InitROS() {
  ROS_INFO_STREAM("Advertising the service and publisher/subscriber to the ROS master");

  // Now setup the connections for the node
  ros::NodeHandle handle;

  // Setup a ROS service that handles requests
  service_ = handle.advertiseService("detect_object_service",
                                     &CascadePartsDetector::HandleServiceRequest,
                                     this);

  // Setup a listener for Image messages
  subscriber_ = handle.subscribe<sensor_msgs::Image, CascadePartsDetector>(
    "detect_object",
    10,
    &CascadePartsDetector::HandleRequest,
    this);

  // Setup the publisher to respond to Image messages
  publisher_ = handle.advertise<DetectionArray>("object_detected", 10, false);

  if (doTiming_) {
    timePublisher_ = handle.advertise<std_msgs::Duration>("processing_time",
                                                          10,
                                                          false);
  }

  return true;
}

void CascadePartsDetector::HandleRequest(const Image::ConstPtr& msg) {
  DetectionArray::Ptr response(new DetectionArray());
  if (HandleRequestImpl(*msg, response.get())) {
    publisher_.publish(response);
  }
}

string CascadePartsDetector::FindPackageDir() {
  string result = "";

  FILE* pipe = popen("rospack find cascade_parts_detector", "r");
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

bool CascadePartsDetector::HandleRequestImpl(const Image& image,
                                             DetectionArray* response) {
  ROS_ASSERT(matlabEngine_);
  ROS_ASSERT(response);

  response->header.seq = image.header.seq;
  response->header.stamp = image.header.stamp;
  response->header.frame_id = image.header.frame_id;

  // Remove old reponse from the matlab engine
  engEvalString(matlabEngine_, "clear bbox");

  // Parse the image encoding
  bool swapRB = false;
  if (image.encoding == "rgb8") {
    swapRB = false;
  } else if (image.encoding == "bgr8") {
    swapRB = true;
  } else {
    ROS_ERROR_STREAM("We can only handle rgb8 or bgr8 enconding, but we got: "
                     << image.encoding);
    return true;
  }

  // Create the matlab image object
  mwSize dims[3];
  dims[0] = image.height;
  dims[1] = image.width;
  dims[2] = 3;
  mxArray* matlabImage = mxCreateNumericArray(3, dims, mxUINT8_CLASS, mxREAL);
  uint8_t* matImagePtr = (uint8_t*)mxGetData(matlabImage);
  const uint8_t* msgImagePtr = &image.data[0];
  for (unsigned int row = 0u; row < image.height; ++row) {
    for (unsigned int col = 0u; col < image.width; ++col) {
      matImagePtr[col*dims[0] + row] = swapRB ? msgImagePtr[2] : msgImagePtr[0];
      matImagePtr[dims[0] * (dims[1] + col) + row] = msgImagePtr[1];
      matImagePtr[dims[0] * (2 * dims[1] + col) + row] = 
        swapRB ? msgImagePtr[0] : msgImagePtr[2];
      msgImagePtr += 3;
    }
  }

  // Send the image to the matlab engine
  engPutVariable(matlabEngine_, "im", matlabImage);
  mxDestroyArray(matlabImage);

  // Start the timer
  Time startTime;
  if (doTiming_) startTime = Time::now();

  // Find the objects
  string findObjCmd;
  if (doCascade_) {
    findObjCmd = ("bbox = cascade_process(im, csc_model);");
  } else {
    findObjCmd = ("bbox = process(im, model, " + thresh_ + ");");
  }
  if (engEvalString(matlabEngine_, findObjCmd.c_str())) {
    ROS_ERROR_STREAM("Error finding the objects in the image:"
                     << matlabBuffer_
                     << " Restarting MATLAB");
    CloseMatlab();
    return InitMatlab();
  }

  if (doTiming_) {
    std_msgs::Duration timeMsg;
    timeMsg.data = Time::now() - startTime;
    timePublisher_.publish(timeMsg);
  }

  // Get the bounding boxes back from Matlab
  mxArray* bboxesMat = engGetVariable(matlabEngine_, "bbox");
  if (bboxesMat) {
  
    // The bbox array will be a 6 column array where the first four
    // columns are (x1, y1, x2, y2). Col 5 is the model component used
    // and 6 is the score.
    int nBoxes = mxGetM(bboxesMat);
    int nCols = mxGetN(bboxesMat);
    if (nBoxes > 0 && nCols < 5) {
      ROS_ERROR_STREAM("The bounding box array is the wrong size: " 
                       << mxGetN(bboxesMat));
      return true;
    }

    // Convert each bounding box into a region of interest message
    double* boxPtr = (double*)mxGetData(bboxesMat);
    for (int i = 0; i < nBoxes; ++i) {
      RegionOfInterest roi;
      roi.x_offset = boxPtr[i];
      roi.y_offset = boxPtr[nBoxes + i];
      roi.height = boxPtr[3*nBoxes + i] - boxPtr[nBoxes + i];
      roi.width = boxPtr[2*nBoxes + i] - boxPtr[i];
      response->detections.push_back(roi);
      response->scores.push_back(boxPtr[(nCols-1)*nBoxes + i]);
    }
    mxDestroyArray(bboxesMat);
  } else {
    ROS_ERROR_STREAM("Could not find the bbox variable. "
                     << "Odds are that there was a matlab error: "
                     << matlabBuffer_);
  }
  return true;
}

}; // cascade_parts_detector
