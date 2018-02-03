// A HOG detector built on integral histograms so that they are fast
// to compute
//
// Copyright 2012 Mark Desnoyer (mdesnoyer@gmail.com

#ifndef __HOG_DETECTOR_INTEGRAL_HOG_DETECTOR_H__
#define __HOG_DETECTOR_INTEGRAL_HOG_DETECTOR_H__

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>
#include <string>
#include <limits.h>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include "cv_utils/IntegralHistogram.h"

namespace hog_detector {

class HogBlockCache;

class HogBlockIterator;

class IntegralHogDescriptorGenerator {
public:
  // Constructor
  //
  // Inputs:
  // winSize - The baseline window size to compute the descriptor on
  // blockSize - The size of the blocks. Must be an integer multiple of cellSize
  // blockStride - The stride of the blocks
  // cellSize - The size of a cell
  // nbins - Number of bins in the histogram. 0-180 degrees unsigned gradients
  IntegralHogDescriptorGenerator(const cv::Size& winSize,
                                 const cv::Size& blockSize,
                                 const cv::Size& blockStride,
                                 const cv::Size& cellSize,
                                 int nbins);

  // Functions to save and load this object to a file
  IntegralHogDescriptorGenerator() {}
  IntegralHogDescriptorGenerator(const std::string& filename);
  bool load(const std::string& filename);
  void save(const std::string& filename) const;
  bool read(const cv::FileNode& node);
  void write(cv::FileStorage& fs) const;

  // Computes the HOG descriptor on a window
  //
  // Inputs:
  // hist - The integral histogram needed to compute the descriptor
  // histSum - An integral image that is the sum of all the histogram bins.
  //           Used for L-1 normalization
  // win - Window to compute the descriptor on. Each dimension should be a 
  //       the same multiple  of the winSize that this object was created with.
  // winStride - The stride in the cannonical image size for the candidate
  //             windows
  //
  // Outputs:
  // descriptor - The resulting hog descriptor. Will be resized if necessary.
  void compute(const cv_utils::IntegralHistogram<float>& hist,
               const cv::Mat_<float>& histSum,
               const cv::Rect& win,
               const cv::Size& winStride,
               std::vector<float>* descriptor) const;

  // Creates an iterator that will iterate through the blocks of the
  // descriptor. Iterator becomes invalid if another is asked for or
  // if compute is called.
  HogBlockIterator CreateBlockIterator(
    const cv_utils::IntegralHistogram<float>& hist,
    const cv::Mat_<float>& histSum,
    const cv::Rect& win,
    const cv::Size& winStride) const;

  int GetDescriptorSize() const { return descriptorSize_; }

  // Getters
  const cv::Size& winSize() const { return winSize_; }
  const cv::Size& blockSize() const { return blockSize_; }
  const cv::Size& blockStride() const { return blockStride_; }
  const cv::Size& cellSize() const { return cellSize_; }
  int descriptorSize() const { return descriptorSize_; }
  int nbins() const { return nbins_; }

private:
  cv::Size winSize_;
  cv::Size blockSize_;
  cv::Size blockStride_;
  cv::Size cellSize_;
  int descriptorSize_;
  int nbins_;

  mutable boost::scoped_ptr<HogBlockCache> cache_;

  // Evil operator
  IntegralHogDescriptorGenerator& operator=(const IntegralHogDescriptorGenerator&);
};

// A class that can compute the hog svm matches more quickly than the
// opencv CVM class because the kernel is prebaked into the decision
// function (and thus linear kernel is necessary).
class HogSVM : public cv::SVM {
public:
  HogSVM() {}
  ~HogSVM();

  bool read(const cv::FileNode& node);
  void write(cv::FileStorage& fs) const;

  virtual bool train(const cv::Mat& trainData, const cv::Mat& responses,
                     const cv::Mat& varIdx=cv::Mat(),
                     const cv::Mat& sampleIdx=cv::Mat(),
                     CvSVMParams params=CvSVMParams());

  virtual float predict(const cv::Mat& sample, bool returnDFVal=false) const;

  template<typename T>
  T predict(HogBlockIterator* blockIter, bool returnDFVal=false) const;

  void SetDecisionVector(float bias, const std::vector<float>& decisionVec,
                         int svmType=EPS_SVR);

private:
  // Vector composed of alpha dot support_vec
  std::vector<float> detectorVec_;

  float bias_;

  template<typename T>
  T predictImpl(const cv::Mat& sample, bool returnDFVal=false) const;
};

class IntegralHogDetector {
public:
  typedef boost::shared_ptr<IntegralHogDetector> Ptr;

  IntegralHogDetector() {};
  IntegralHogDetector(const std::string& filename,
                      const cv::Size& winStride);
  IntegralHogDetector(const cv::Size& winSize,
                      const cv::Size& blockSize,
                      const cv::Size& blockStride,
                      const cv::Size& cellSize,
                      int nbins,
                      float thresh=-std::numeric_limits<float>::infinity(),
                      const cv::Rect& subWindow=cv::Rect());

  bool load(const std::string& filename);
  void save(const std::string& filename) const;
  bool read(const cv::FileNode& node);
  void write(cv::FileStorage& fs) const;

  // Train the detector from a bunch of images. Takes the center
  // winSize portion of the image for training. Typical parameters
  // are: linear kernel, c=0.01, regression, ep=0.1
  //
  // Inputs:
  // imageFiles - List of image files to train with
  // labels - List of labels for said image files
  void Train(const std::vector<std::string>& imageFiles,
             const std::vector<float>& labels);

  // Add some regions to a set of training regions
  void AddRegionsForTraining(const cv::Mat& image,
                             const std::vector<cv::Rect>& rois,
                             const std::vector<float>& labels,
                             bool addToFront=false);

  // After running AddRegions, call this to do the training.
  void DoTraining();

  void ClearTrainingData() {
    trainEntries_.clear();
    trainLabels_.clear();
  }

  // Detect objects in an image
  // Inputs:
  // image - image to look for objects in
  // roisIn - rectangles to evaluate
  // hist - Precomupted integral histogram for this image (optional)
  // histSum - Precomputed integral image for the histogram energy (optional)
  //
  // Outputs:
  // foundLocations - rectangles where the objects were found
  // scores - score for each of the objects found
  // processingTime - Processing time to find the objects (optional)
  bool DetectObjects(const cv::Mat& image,
                     const std::vector<cv::Rect>& roisIn,
                     std::vector<cv::Rect>* foundLocations,
                     std::vector<double>* scores,
                     double* processingTime=NULL,
                     const cv_utils::IntegralHistogram<float>* hist=NULL,
                     const cv::Mat_<float>* histSum=NULL) const;

  // Computes the score of a window
  //
  // Inputs:
  // hist - Precomupted integral histogram for this image
  // histSum - Precomputed integral image for the histogram energy
  // roi - Region of interest to compute the score of
  //
  // Outputs:
  // return - The score of the window
  float ComputeScore(const cv_utils::IntegralHistogram<float>& hist,
                     const cv::Mat_<float>& histSum,
                     const cv::Rect& roi) const;

  // From an image, computes the integral histogram needed to compute
  // the HOG descriptor.
  //
  // Inputs:
  // image - Image to compute the oriented gradient histogram of
  //
  // Outputs:
  // histSum - Optional. An integral image of the L1 sum in the histogram
  // returns - The integral histogram. Caller must take ownership.
  cv_utils::IntegralHistogram<float>* ComputeGradientIntegralHistograms(
    const cv::Mat& image,
    cv::Mat_<float>* histSum=NULL) const;

  // Getters
  const IntegralHogDescriptorGenerator& generator() const {
    return generator_;
  }
  const cv::Size& winSize() const { return winSize_; }
  const cv::Size& blockSize() const { return generator_.blockSize(); }
  const cv::Rect& subWindow() const { return subWindow_; }
  int descriptorSize() const { return generator_.descriptorSize(); }
  float thresh() const { return thresh_; }
  int nbins() const { return nbins_; }

  // Setters
  void SetWinStride(const cv::Size stride) {winStride_ = stride;}
  void SetThresh(float thresh) {thresh_ = thresh;}

private:
  // The svm model for this detector
  HogSVM svm_;

  // The object to generate the actual descriptors
  IntegralHogDescriptorGenerator generator_;

  // The cannonical window size
  cv::Size winSize_;

  // Instead of computing the feature on the whole window asked, this
  // specifies the sub-window to use in coordinates of the cannonical window.
  cv::Rect subWindow_;

  // The threshold to accept the score when returning objects
  float thresh_;
  
  // Number of histogram bins
  int nbins_;

  // The stride for windows to be requested. Needed to cache properly
  cv::Size winStride_;

  // Objects for holding training data. These will be empty once
  // DoTraining is called.
  typedef std::vector<boost::shared_ptr<std::vector<float> > > DescriptorList;
  DescriptorList trainEntries_;
  std::vector<float> trainLabels_;

  // Computes the gradients of the image at every point. Angles are
  // converted to the unsigned format and are thus 0-pi.
  void ComputeGradients(const cv::Mat& image,
                        cv::Mat_<float>* magnitude,
                        cv::Mat_<float>* angle) const;

  // Converts a requested window into the subwindow where we actually
  // compute the hog descriptor.
  cv::Rect_<double> Win2SubWin(const cv::Rect& win) const;
};

class HogBlockCache {
public:
  HogBlockCache(int nbins,
                const cv::Size& winStride,
                const cv::Size& blockStride);

  void InitIfNecessary(const cv_utils::IntegralHistogram<float>* hist,
                       const cv::Mat_<float>* histSum,
                       float xFactor,
                       float yFactor,
                       const cv::Size& blockSize,
                       const cv::Size& cellSize);

  // Retrieves the block whose upper left corner is at (blockX,
  // blockY) inside the window win.
  inline const float* GetBlock(const cv::Rect& win, int blockX, int blockY) const;

  int blockHistSize() const { return blockHistSize_; }

private:
  int cellHistSize_;
  cv::Size cacheStride_;

  const cv_utils::IntegralHistogram<float>* hist_;
  int histId_;
  const cv::Mat_<float>* histSum_;
  float xFactor_;
  float yFactor_;
  cv::Size blockSize_;
  cv::Size cellSize_;
  int blockHistSize_;

  mutable cv::Mat_<float> blockCache_;
  mutable cv::Mat_<uchar> flags_;

  const float* CalculateBlock(const cv::Rect& win, int blockX, int blockY,
                              float* blockPtr) const ;

  // Evil constructors
  HogBlockCache(const HogBlockCache&);
  HogBlockCache& operator=(const HogBlockCache&);
};

// Iterator that returns the hog blocks one at a time
class HogBlockIterator {
public:
  HogBlockIterator(const cv::Rect& win, const HogBlockCache* cache,
                   const cv::Size& winSize, const cv::Size& blockSize,
                   const cv::Size& blockStride) 
    : win_(win), cache_(cache), winSize_(winSize), blockSize_(blockSize),
      blockStride_(blockStride), curBlockX_(0), curBlockY_(0),
      blockHistSize_(cache_->blockHistSize()), isDone_(false) {}

  inline const float* operator*() const;

  // Only the prefix operator is defined
  HogBlockIterator& operator++() {
    curBlockX_ += blockStride_.width;
    if (curBlockX_ + blockSize_.width > winSize_.width) {
      curBlockX_ = 0;
      curBlockY_ += blockStride_.height;
      if (curBlockY_ + blockSize_.height > winSize_.height) {
        isDone_ = true;
      }
    }
    return *this;
  }

  bool isDone() { return isDone_;}

  int blockHistSize() { return blockHistSize_; }

private:
  const cv::Rect& win_;
  const HogBlockCache* cache_;
  const cv::Size& winSize_;
  const cv::Size& blockSize_;
  const cv::Size& blockStride_;
  int curBlockX_;
  int curBlockY_;
  int blockHistSize_;
  bool isDone_;
};
  

} // namespace

#endif // __HOG_DETECTOR_INTEGRAL_HOG_DETECTOR_H__
