// A HOG detector built on integral histograms so that they are fast
// to compute
//
// Copyright 2012 Mark Desnoyer (mdesnoyer@gmail.com)

#include <ros/ros.h>
#include "hog_detector/integral_hog_detector_inl.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/math/special_functions/round.hpp>
#include <math.h>
#include <algorithm>
#include "cv_utils/IntegralHistogram-Inl.h"
#include "cv_utils/math.h"
#include "cv_utils/DisplayImages.h"

using namespace cv;
using namespace std;
using boost::scoped_ptr;
using boost::shared_ptr;
using cv_utils::IntegralHistogram;

namespace hog_detector {

inline float round(float val) {
  return static_cast<int>(val + 0.5);
}

IntegralHogDescriptorGenerator::IntegralHogDescriptorGenerator(
  const string& filename) {
  load(filename);
}

bool IntegralHogDescriptorGenerator::load(const string& filename) {
  FileStorage fs(filename, FileStorage::READ);
  return read(fs.getFirstTopLevelNode());
}

void IntegralHogDescriptorGenerator::save(const string& filename) const {
  FileStorage fs(filename, FileStorage::WRITE);
  fs << FileStorage::getDefaultObjectName(filename);
  return write(fs);
}

bool IntegralHogDescriptorGenerator::read(const cv::FileNode& node) {
  if (!node.isMap()) {
    return false;
  }
  FileNodeIterator it = node["winSize_"].begin();
  it >> winSize_.width >> winSize_.height;
  it = node["blockSize_"].begin();
  it >> blockSize_.width >> blockSize_.height;
  it = node["blockStride_"].begin();
  it >> blockStride_.width >> blockStride_.height;
  it = node["cellSize_"].begin();
  it >> cellSize_.width >> cellSize_.height;
  node["descriptorSize_"] >> descriptorSize_;
  node["nbins_"] >> nbins_;

  return true;
}

void IntegralHogDescriptorGenerator::write(cv::FileStorage& fs) const {
  fs << "{:" "md-integral-hog-descriptor-generator"
     << "winSize_" << winSize_
     << "blockSize_" << blockSize_
     << "blockStride_" << blockStride_
     << "cellSize_" << cellSize_
     << "descriptorSize_" << descriptorSize_
     << "nbins_" << nbins_;
  fs << "}";
}

IntegralHogDescriptorGenerator::IntegralHogDescriptorGenerator(
  const cv::Size& winSize,
  const cv::Size& blockSize,
  const cv::Size& blockStride,
  const cv::Size& cellSize,
  int nbins) 
  : winSize_(winSize), blockSize_(blockSize), blockStride_(blockStride),
    cellSize_(cellSize), descriptorSize_(0), nbins_(nbins),
    cache_(NULL) {
  // Make sure all the sizes will work nicely
  ROS_ASSERT(blockSize.width % cellSize.width == 0 &&
             blockSize.height % cellSize.height == 0);
  ROS_ASSERT(winSize.width % blockSize.width == 0 &&
             winSize.height % blockSize.height == 0);
  ROS_ASSERT(winSize.width % blockStride.width == 0 &&
             winSize.height % blockStride.height == 0);

  descriptorSize_ = nbins * 
    (blockSize.width / cellSize.width) *
    (blockSize.height / cellSize.height) *
    ((winSize.width - blockSize.width) / blockStride.width + 1) *
    ((winSize.height - blockSize.height) / blockStride.height + 1);
}

void IntegralHogDescriptorGenerator::compute(
  const IntegralHistogram<float>& hist,
  const Mat_<float>& histSum,
  const Rect& win,
  const cv::Size& winStride,
  vector<float>* descriptor) const {
  ROS_ASSERT(descriptor != NULL);

  descriptor->reserve(descriptorSize_);
  descriptor->resize(descriptorSize_);
  float* descriptorPtr = &((*descriptor)[0]);

  const float xFactor = static_cast<float>(win.width) / winSize_.width;
  const float yFactor = static_cast<float>(win.height) / winSize_.height;

  if (cache_.get() == NULL) {
    cache_.reset(new HogBlockCache(nbins_, winStride, blockStride_));
  }

  cache_->InitIfNecessary(&hist, &histSum, xFactor, yFactor,
                          blockSize_, cellSize_);

  int blockHistSize = cache_->blockHistSize();
  
  // Build up the descriptor one block at a time
  int blockW = blockSize_.width;
  int blockH = blockSize_.height;
  for (int blockY = 0; blockY + blockH <= winSize_.height;
       blockY += blockStride_.height) {
    for (int blockX = 0; blockX + blockW <= winSize_.width;
         blockX += blockStride_.width) {
      const float* blockHist = cache_->GetBlock(win, blockX, blockY);

      std::copy(blockHist, blockHist + blockHistSize, descriptorPtr);
      
      descriptorPtr += blockHistSize;
    }
  }
  // Make sure we counted the size of the descriptor properly
  ROS_ASSERT(descriptorPtr - &((*descriptor)[0]) == descriptorSize_);
}

HogBlockIterator IntegralHogDescriptorGenerator::CreateBlockIterator(
    const cv_utils::IntegralHistogram<float>& hist,
    const cv::Mat_<float>& histSum,
    const cv::Rect& win,
    const cv::Size& winStride) const {
  const float xFactor = static_cast<float>(win.width) / winSize_.width;
  const float yFactor = static_cast<float>(win.height) / winSize_.height;

  if (cache_.get() == NULL) {
    cache_.reset(new HogBlockCache(nbins_, winStride, blockStride_));
  }

  cache_->InitIfNecessary(&hist, &histSum, xFactor, yFactor,
                          blockSize_, cellSize_);

  return HogBlockIterator(win, cache_.get(), winSize_, blockSize_,
                          blockStride_);
}

/*
void IntegralHogDescriptorGenerator::compute(
  const IntegralHistogram<float>& hist,
  const Mat_<float>& histSum,
  const Rect& win,
  const cv::Size& winStride,
  vector<float>* descriptor) const {
  ROS_ASSERT(descriptor != NULL);

  descriptor->reserve(descriptorSize_);
  descriptor->resize(descriptorSize_);
  float* descriptorPtr = &((*descriptor)[0]);

  const float xFactor = static_cast<float>(win.width) / winSize_.width;
  const float yFactor = static_cast<float>(win.height) / winSize_.height;

  // Loop through the block locations
  const unsigned int curHistSize = hist.nbins();

  // Store some constants to avoid lookups deep in the loops
  int blockW = blockSize_.width;
  int blockH = blockSize_.height;
  int cellW = cellSize_.width;
  int cellH = cellSize_.height;
  int winX = win.x;
  int winY = win.y;
  int cellX;

  float* blockHistStart = &((*descriptor)[0]);
  for (int blockY = 0; blockY + blockH <= winSize_.height;
       blockY += blockStride_.height) {
    for (int blockX = 0; blockX + blockW <= winSize_.width;
       blockX += blockStride_.width) {
      // Grab the region spanned by the block
      const int curX = round(winX + blockX*xFactor);
      const int curY = round(winY + blockY*yFactor);
      const int curW = round(blockW*xFactor);
      const int curH = round(blockH*yFactor);

      // Get the normalization factor for the block
      float normFactor = histSum(curY, curX) -
        histSum(curY, curX + curW) +
        histSum(curY + curH, curX + curW) -
        histSum(curY + curH, curX);
      if (normFactor > 1e-16) {
        normFactor = 1.f / normFactor;
      } else {
        normFactor = 0;
      }

      // Go through the cells and calculate the histograms
      for (int cellY = 0; cellY < blockH; cellY+=cellH) {
        for(cellX = 0; cellX < blockW; cellX += cellW) {
          hist.GetHistInRegion(round(winX + xFactor*(blockX + cellX)),
                               round(winY + yFactor*(blockY + cellY)),
                               round(xFactor*cellW),
                               round(yFactor*cellH),
                               blockHistStart,
                               normFactor);
          blockHistStart += curHistSize;
        }
      }
    }
  }
  // Make sure we counted the size of the descriptor properly
  ROS_ASSERT(blockHistStart - &((*descriptor)[0]) == descriptorSize_);
}
*/

IntegralHogDetector::IntegralHogDetector(
const string& filename, const cv::Size& winStride) {
  load(filename);  
  winStride_ = winStride;
}

IntegralHogDetector::IntegralHogDetector(const cv::Size& winSize,
                                         const cv::Size& blockSize,
                                         const cv::Size& blockStride,
                                         const cv::Size& cellSize,
                                         int nbins,
                                         float thresh,
                                         const cv::Rect& subWindow)
  : svm_(),
    generator_(subWindow.area() == 0 ? winSize : subWindow.size(),
               blockSize,
               blockStride,
               cellSize,
               nbins),
    winSize_(winSize),
    subWindow_(subWindow),
    thresh_(thresh),
    nbins_(nbins) {
  // Do a trivial training on the svm so that it contains valid data
  Mat_<float> trainData = (Mat_<float>(2,2) << 0, 0, 1, 1);
  Mat_<float> labels = (Mat_<float>(2, 1) << 0, 1);
  CvSVMParams params;
  params.svm_type = CvSVM::EPS_SVR;
  params.kernel_type = CvSVM::LINEAR;
  params.C = 0.01;
  params.p = 0.1;
  svm_.train(trainData, labels, Mat(), Mat(), params);
}

bool IntegralHogDetector::load(const string& filename) {
  FileStorage fs(filename, FileStorage::READ);
  return read(fs.getFirstTopLevelNode());
}
 
void IntegralHogDetector::save(const string& filename) const {
  FileStorage fs(filename, FileStorage::WRITE);
  fs << FileStorage::getDefaultObjectName(filename);
  return write(fs);
}

bool IntegralHogDetector::read(const cv::FileNode& node) {
  if (!node.isMap()) {
    return false;
  }
  FileNode curNode;
  curNode = node["generator_"];
  if (!generator_.read(curNode)) {
    return false;
  }
  
  FileNodeIterator it = node["winSize_"].begin();
  it >> winSize_.width >> winSize_.height;
  it = node["subWindow_"].begin();
  it >> subWindow_.x >> subWindow_.y >> subWindow_.width >> subWindow_.height;
  node["thresh_"] >> thresh_;
  node["nbins_"] >> nbins_;
  
  curNode = node["svm_"];
  return svm_.read(curNode); 
}

void IntegralHogDetector::write(cv::FileStorage& fs) const {
  fs << "{:" "md-integral-hog-detector";

  fs << "generator_";
  generator_.write(fs);
  fs << "winSize_" << winSize_
     << "subWindow_" << subWindow_
     << "thresh_" << thresh_
     << "nbins_" << nbins_;
  
  fs << "svm_";
  svm_.write(fs);
  fs << "}";
}

void IntegralHogDetector::Train(const vector<string>& imageFiles,
                                const vector<float>& labels) {
  ROS_ASSERT(labels.size() == imageFiles.size());

  // Now, build up our list of descriptors to train with
  int curRow = 0;
  for (vector<string>::const_iterator imageFile = imageFiles.begin();
       imageFile != imageFiles.end(); ++imageFile) {
    ROS_INFO_STREAM("Extracting HOG descriptor from " << *imageFile);

    // Start by opening the image
    Mat image = imread(*imageFile);

    // Find the winSize_ window in the center of the image
    cv::Rect win((image.cols - winSize_.width) / 2,
                 (image.rows - winSize_.height) / 2,
                 winSize_.width,
                 winSize_.height);

    AddRegionsForTraining(image, vector<Rect>(1, win),
                          vector<float>(1, labels[curRow]));

    curRow++;
  }

  DoTraining();
}

void IntegralHogDetector::AddRegionsForTraining(
  const cv::Mat& image,
  const std::vector<Rect>& rois,
  const std::vector<float>& labels,
  bool addToFront) {
  ROS_ASSERT(rois.size() == labels.size());

  // Calculate the integral histograms for the image
  Mat_<float> histSum;
  scoped_ptr<IntegralHistogram<float> > hist(
    ComputeGradientIntegralHistograms(image, &histSum));

  DescriptorList curEntries;
  for (vector<Rect>::const_iterator roiI = rois.begin();
       roiI != rois.end(); ++roiI) {
    curEntries.push_back(shared_ptr<vector<float> >(new vector<float>()));
    generator_.compute(*hist, histSum, Win2SubWin(*roiI), winStride_,
                       curEntries.back().get());
  }

  // Load up the labels
  if (addToFront) {
    trainLabels_.insert(trainLabels_.begin(), labels.begin(), labels.end());
    trainEntries_.insert(trainEntries_.begin(), curEntries.begin(),
                         curEntries.end());
  } else {
    trainLabels_.insert(trainLabels_.end(), labels.begin(), labels.end());
    trainEntries_.insert(trainEntries_.end(), curEntries.begin(),
                         curEntries.end());
  }
}

void IntegralHogDetector::DoTraining() {
  ROS_ASSERT(generator_.GetDescriptorSize() == trainEntries_[0]->size());
  const int descriptorSize = generator_.GetDescriptorSize();

  svm_.clear();

  // Setup the parameters for training first
  CvSVMParams params;
  //params.svm_type = CvSVM::EPS_SVR;
  params.svm_type = CvSVM::C_SVC;
  params.kernel_type = CvSVM::LINEAR;
  params.C = 0.01;
  params.p = 0.1;

  // Make a matrix of the training examples
  Mat_<float> trainMat(trainEntries_.size(), descriptorSize);
  for (unsigned int i = 0u; i < trainEntries_.size(); ++i) {
    std::copy(trainEntries_[i]->begin(), trainEntries_[i]->end(),
              trainMat.ptr<float>(i));
  }

  ROS_INFO_STREAM("Starting the training using " << trainLabels_.size()
                  << " examples");
  svm_.train(trainMat, Mat_<float>(trainLabels_), Mat(), Mat(), params);
  ROS_INFO("Done training");
}

bool IntegralHogDetector::DetectObjects(
  const cv::Mat& image,
  const std::vector<cv::Rect>& roisIn,
  std::vector<cv::Rect>* foundLocations,
  std::vector<double>* scores,
  double* processingTime,
  const IntegralHistogram<float>* hist,
  const cv::Mat_<float>* histSum) const {
  ROS_ASSERT(scores && foundLocations);

  // Start the timer
  ros::Time startTime;
  if (processingTime) startTime = ros::Time::now();

  // Compute the integral histograms if necessary
  const IntegralHistogram<float>* histPtr = hist;
  const cv::Mat_<float>* histSumPtr = histSum;
  scoped_ptr<IntegralHistogram<float> > histImpl;
  cv::Mat_<float> histSumImpl;
  if (hist == NULL || histSum == NULL) {
    histImpl.reset(ComputeGradientIntegralHistograms(image,
                                                     &histSumImpl));
    histPtr = histImpl.get();
    histSumPtr = &histSumImpl;
  }

  // Now march through the regions of interest and calculate the scores
  for (vector<Rect>::const_iterator roiI = roisIn.begin();
       roiI != roisIn.end(); ++roiI) {
    float curScore = ComputeScore(*histPtr, *histSumPtr, *roiI);

    // If the score is good enough, set for output
    if (curScore > thresh_) {
      scores->push_back(curScore);
      foundLocations->push_back(*roiI);
    }
  }

  // Stop the timer
  if (processingTime) {
    *processingTime = (ros::Time::now() - startTime).toSec();
  }

  return true;
}

float IntegralHogDetector::ComputeScore(
  const cv_utils::IntegralHistogram<float>& hist,
  const cv::Mat_<float>& histSum,
  const cv::Rect& roi) const {
  
  // Get the descriptor for this region
  HogBlockIterator blockIter = generator_.CreateBlockIterator(hist, histSum,
                                                              Win2SubWin(roi),
                                                              winStride_);

  // Score the region
  float score = svm_.predict<float>(&blockIter, true);

  return score;
}

IntegralHistogram<float>* 
IntegralHogDetector::ComputeGradientIntegralHistograms(
  const cv::Mat& image, cv::Mat_<float>* histSum) const {

  Mat_<float> magnitude;
  Mat_<float> angle;
  ComputeGradients(image, &magnitude, &angle);

  integral(magnitude, *histSum, CV_32F);

  return IntegralHistogram<float>::Calculate<float>(
    angle,
    nbins_,
    &pair<float, float>(0, M_PI),
    magnitude,
    IntegralHistogram<float>::LINEAR_INTERP);
}

void IntegralHogDetector::ComputeGradients(const cv::Mat& image,
                                           cv::Mat_<float>* magnitude,
                                           cv::Mat_<float>* angle) const {
  ROS_ASSERT(magnitude && angle);

  // First convert to greyscale
  const Mat* greyImage = &image;
  Mat greyImageTmp;
  if (image.channels() != 1) {
    cvtColor(image, greyImageTmp, CV_BGR2GRAY);
    greyImage = &greyImageTmp;
  }  

  // Compute the gradients in the x and y directions
  Mat_<float> xGrad;
  Mat_<float> yGrad;
  Mat_<float> derivKernel = (Mat_<float>(3, 1) << -1, 0, 1);
  Mat_<float> emptyKernel = (Mat_<float>(1, 1) << 1);
  sepFilter2D(*greyImage, xGrad, xGrad.type(), derivKernel, emptyKernel);
  sepFilter2D(*greyImage, yGrad, yGrad.type(), emptyKernel, derivKernel);
  

  // Convert to angle and magnitude
  cartToPolar(xGrad, yGrad, *magnitude, *angle);

  // Normalize the graident
  if (greyImage->depth() == CV_8U) {
    *magnitude /= 255.0;
  }

  // Make the angles unsigned
  Mat_<float> flipAngle = *angle > M_PI;
  *angle -= flipAngle * (M_PI / 255.0);
}

cv::Rect_<double> IntegralHogDetector::Win2SubWin(const cv::Rect& win) const {
  if (subWindow_.area() == 0) {
    return win;
  }

  const double xFactor = static_cast<double>(win.width) / winSize_.width;
  const double yFactor = static_cast<double>(win.height) / winSize_.height;

  return Rect_<double>(win.x + xFactor*subWindow_.x,
                       win.y + yFactor*subWindow_.y,
                       xFactor * subWindow_.width,
                       yFactor * subWindow_.height);
              
}

HogSVM::~HogSVM() {}

bool HogSVM::read(const cv::FileNode& node) {
  if (!node.isMap()) {
    return false;
  }

  detectorVec_.clear();
  FileNode curNode = node["detectorVec_"];
  for (FileNodeIterator i = curNode.begin(); i != curNode.end(); ++i) {
    float curVal;
    (*i) >> curVal;
    detectorVec_.push_back(curVal);
  }

  node["bias_"] >> bias_;

  return true;
}

void HogSVM::write(cv::FileStorage& fs) const {
  fs << "{:" "md-hog-svm";
  fs << "detectorVec_" << "[:";
  for (vector<float>::const_iterator i = detectorVec_.begin();
       i != detectorVec_.end(); ++i) {
    fs << *i;
  }
  fs << "]";

  fs << "bias_" << bias_;
  fs << "}";
}


bool HogSVM::train(const cv::Mat& trainData,
                   const cv::Mat& responses,
                   const cv::Mat& varIdx,
                   const cv::Mat& sampleIdx,
                   CvSVMParams params) {
  ROS_ASSERT(params.kernel_type == CvSVM::LINEAR);
  //ROS_ASSERT(params.svm_type == EPS_SVR || params.svm_type == NU_SVR ||
  //           params.svm_type == ONE_CLASS);

  // First train the svm model normally
  if (!SVM::train(trainData, responses, varIdx, sampleIdx, params)) {
    return false;
  }

  // Now, build the detector vector using the trained results. This
  // will be created such that det[i] = sum_j(sv_j[i]*alpha[j]) where
  // j is the number of support vectors and i is the descriptor
  // size. Also, keep track of the bias
  bias_ = -decision_func->rho;
  int sv_count = decision_func->sv_count;
  int var_count = get_var_count();
  detectorVec_.resize(var_count);
  for (int i = 0; i < var_count; ++i) {
    float sum = 0;
    for (int j = 0; j < sv_count; ++j) {
      sum += decision_func->alpha[j]*sv[j][i];
    }
    detectorVec_[i] = sum;
  }

  return true;
}

template<typename T>
T HogSVM::predictImpl(const cv::Mat& sample,
                      bool returnDFVal) const {
  T score = bias_;

  ROS_ASSERT(sample.rows == (int)detectorVec_.size() ||
             sample.cols == (int)detectorVec_.size());

  if (sample.isContinuous()) {
    const T* samplePtr = sample.ptr<T>(0);
    const float* vectorPtr = &detectorVec_[0];
    const unsigned int vecSize = detectorVec_.size();
    unsigned int i = 0;
    for (; i <= vecSize - 4; i+=4 ) {
      score += samplePtr[i]*vectorPtr[i] + 
        samplePtr[i+1]*vectorPtr[i+1] + 
        samplePtr[i+2]*vectorPtr[i+2] + 
        samplePtr[i+3]*vectorPtr[i+3];
    }
    for (; i < vecSize; ++i) {
      score += samplePtr[i]*vectorPtr[i];
    }
  } else {
    ROS_FATAL("Descriptor must be continuous for the moment");
    ROS_ASSERT(false);
  }

  return returnDFVal ? score : static_cast<T>(score > 0);
}

float HogSVM::predict(const cv::Mat& sample,
                      bool returnDFVal) const {
  ROS_ASSERT(sample.channels() == 1);

  if (sample.depth() == CV_32F) {
    return predictImpl<float>(sample, returnDFVal);
  } else if (sample.depth() == CV_64F) {
    return predictImpl<double>(sample, returnDFVal);
  } else if (sample.depth() == CV_32S) {
    return predictImpl<int32_t>(sample, returnDFVal);
  } else if (sample.depth() == CV_8U) {
    return predictImpl<uint8_t>(sample, returnDFVal);
  }
  
  ROS_FATAL("Sample must be 32F, 64F, 32S or 8U");
  ROS_ASSERT(false);
  return 0.0;
}

void HogSVM::SetDecisionVector(float bias,
                               const std::vector<float>& decisionVec,
                               int svmType) {
  bias_ = bias;
  detectorVec_ = decisionVec;
  params.svm_type = svmType;
}

HogBlockCache::HogBlockCache(int nbins, const cv::Size& winStride,
                             const cv::Size& blockStride)
   : cellHistSize_(nbins),
     cacheStride_(),
     hist_(NULL), histId_(-1), histSum_(NULL), xFactor_(0),
    yFactor_(0) {
  if (winStride.width == 0 || winStride.height == 0) {
    cacheStride_ = blockStride;
  } else {
    cacheStride_.width = gcd(winStride.width, blockStride.width);
    cacheStride_.height = gcd(winStride.height, blockStride.height);
  }
}


const float* HogBlockCache::CalculateBlock(const Rect& win, int blockX,
                                           int blockY, float* blockPtr) const {
  const int curX = std::min(static_cast<int>(round(win.x + blockX*xFactor_)),
                            win.x + win.width - blockSize_.width);
  const int curY = std::min(static_cast<int>(round(win.y + blockY*yFactor_)),
                            win.y + win.height - blockSize_.height);

  // Calculate the block
  const int curW = blockSize_.width;
  const int curH = blockSize_.height;

  float normFactor = (*histSum_)(curY, curX) -
    (*histSum_)(curY, curX + curW) +
    (*histSum_)(curY + curH, curX + curW) -
    (*histSum_)(curY + curH, curX);

  if (normFactor > 1e-16) {
    normFactor = 1.f / normFactor;
  } else {
    normFactor = 0;
  }

  const int cellH = cellSize_.height;
  const int cellW = cellSize_.width;

  float* retPtr = blockPtr;

  // Go through the cells and calculate the histograms
  for (int cellY = 0; cellY <= blockSize_.height-cellH; cellY += cellH) {
    for(int cellX = 0; cellX <= blockSize_.width-cellW; cellX += cellW) {
      hist_->GetHistInRegion(curX + cellX,
                             curY + cellY,
                             cellW,
                             cellH,
                             blockPtr,
                             normFactor);
      blockPtr += cellHistSize_;
    }
  }

  return retPtr;
}

void HogBlockCache::InitIfNecessary(
  const cv_utils::IntegralHistogram<float>* hist,
  const cv::Mat_<float>* histSum,
  float xFactor,
  float yFactor,
  const cv::Size& blockSize,
  const cv::Size& cellSize) {
  if (hist_ && hist->randId() == histId_  &&
      fabs(xFactor_-xFactor) < 1e-10) {
    // We're already initialized
    ROS_ASSERT(fabs(yFactor_-yFactor) < 1e-10);
    return;
  }

  // Not initialized for this size window or image, so do so.
  hist_ = hist;
  histId_ = hist->randId();
  histSum_ = histSum;
  xFactor_ = xFactor;
  yFactor_ = yFactor;
  blockSize_ = Size(round(blockSize.width * xFactor),
                    round(blockSize.height * yFactor));
  cellSize_ = Size(round(cellSize.width * xFactor),
                   round(cellSize.height * yFactor));

  // Allocate the cache data if the size has changed
  vector<int> sizes;
  sizes.push_back((histSum_->rows-1) / cacheStride_.height + 1);
  sizes.push_back((histSum_->cols-1) / cacheStride_.width + 1 );
  blockHistSize_ = cellHistSize_ * blockSize.width / cellSize.width *
    blockSize.height / cellSize.height;
  sizes.push_back(blockHistSize_);
  blockCache_.create(3, &(sizes[0]));
  flags_.create(2, &(sizes[0]));

  // Zero all the flags
  flags_ = 0;
}
                       

} // namespace
