// Copyright 2011 Carnegie Mellon University
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
//
// VisualUtilityEstimator.h
//
// An object that will calculate the visual utility for a given image

#ifndef __VISUAL_UTILITY_ESTIMATOR_H__
#define __VISUAL_UTILITY_ESTIMATOR_H__

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>
#include <boost/scoped_ptr.hpp>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include "TransformEstimator.h"
#include "objectness/objectness.h"
#include "hog_detector/hog_detector_internal.h"
#include "cascade_detector/detailed_cascade_detector.h"
#include "cv_utils/IntegralHistogram.h"
#include "hog_detector/integral_hog_detector.h"
#include "hog_detector/integral_hog_cascade.h"
#include <ext/hash_map>
using __gnu_cxx::hash_map;

namespace visual_utility {

// Abstract visual utility estimator
class VisualUtilityEstimator {
public:
  VisualUtilityEstimator() : lastRuntime_(NULL) {}
  virtual ~VisualUtilityEstimator();

  typedef std::pair<double, cv::Rect> ROIScore;

  // Workhorse function that calculates the visual utility for every frame
  virtual
  cv::Mat_<double> CalculateVisualUtility(const cv::Mat& image,
                                          double time)=0;

  // Calculates the visual utility for each region of interest where
  // the regions are boxes and have the same aspect ratio. This is
  // equivalent to caluclating the visual utility at a bunch of
  // different locations and scales.
  //
  // Inputs:
  // image - Image to calculate the visual utility on
  // rois - The regions of interest to calculate the visual utility for
  // time - The time the image was taken
  // vuOut - Output of the visual utility score for each entry in rois.
  virtual void CalculateVisualUtility(const cv::Mat& image,
                                      const std::vector<cv::Rect>& rois,
                                      double time,
                                      std::vector<ROIScore>* vuOut);

  // Calculates the visual utility for each region for an image from a
  // given filename. This is mostly useful where there is some
  // cacheing going on.
  virtual void CalculateVisualUtility(const std::string& filename,
                                      const std::vector<cv::Rect>& rois,
                                      double time,
                                      std::vector<ROIScore>* vuOut);

  // Calculate the visual utility on a grid of bounding boxes. The
  // grid is defined as (w,h,x,y) unless the aspect ratio is fixed and
  // then it becomes (s,x,y)
  //
  // image - Image to calculate visual utility for
  // min[X,Y,W,H] - The minimum x,y,w,h respectively
  // stride[X,Y] - Stride to sample the grid at for the (x,y) locations
  // stride[W,H] - Stride to sample the gride in the height width. w_i = round(w_{i-1}*strideW)
  // fixAspect - If true, only strideW is used and the height goes in lockstep
  // time - Time the image was taken
  // mask - Optional binary mask that is the same shape as the output,
  //        specifying the valid entires to compute.
  virtual cv::Mat CalculateVisualUtility(const cv::Mat& image,
                                         int minX, int minY,
                                         int minW, int minH,
                                         int strideX, int strideY,
                                         double strideW, double strideH,
                                         bool fixAspect,
                                         double time,
                                         const cv::Mat& mask=cv::Mat());
  

  // Some of the visual utility estimators can calculate a transform.
  // If they do, this function will return the transform from the
  // frame 2 back to the last frame that we calculated the visual
  // utility for.
  //
  // If there's no transform that is found, NULL is returned.
  virtual const cv::Mat* GetLastTransform() const;

  // Can be NULL if it is not calculated internally. In that case, you
  // should just measure the time yourself.
  virtual double* GetLastRuntime() const {
    return lastRuntime_.get();
  }

protected:
  typedef std::vector<std::pair<double*, cv::Point> > PointScores;

  // The runtime in seconds of the last visual utility calculation
  mutable boost::scoped_ptr<double> lastRuntime_;

  // When we want to calculate the visual utility of boxes, the
  // default behaviour will first call InitBoxCalculator so that a
  // subclass can do any initializations necessary. Then, calls will
  // be made to CalculateVisualUtilityBoxAcrossImage, which in turn calls
  // CalculateVisualUtilityOfBox. At a minimum, subclasses must
  // implement InitBoxCalculator and CalculateVisualUtilityOfBox.
  virtual bool InitBoxCalculator(const cv::Mat& image, double time);

  // Calculates the visual utility for boxes of size width x height in
  // the desired locations in the image.
  //
  // width/height - Size of the box to move around
  // locations - List of upper left corners to evaluate. Scores will be written to the double* pointer.
  virtual void CalculateVisualUtilityBoxAcrossImage(
    double width, double height,
    const PointScores& locations);
  
  // Calculates the visual utility of a box. Is called after
  // InitBoxCalculator()
  virtual double CalculateVisualUtilityOfBox(int x, int y, int h,
                                             int w);

  // Helper function that lists the widths and heights of boxes
  // to evaluate computed in a grid.
  void GetGridHeightsAndWidths(const cv::Mat& image,
                               int minX, int minY,
                               int minW, int minH, double strideW,
                               double strideH, bool fixAspect,
                               std::vector<double>* widths,
                               std::vector<double>* heights);

  // Creates an initial empty score grid that is output from the grid
  // based calculator. It is the right size and every entry is -inf.
  void InitializeScoreGrid(const cv::Mat& image, int minX, int minY,
                           int strideX, int strideY,
                           bool fixAspect,
                           const std::vector<double>& widths,
                           const std::vector<double>& heights,
                           cv::Mat* scoreGrid);


private:
  void GetLocationsForEvaluation(const cv::Mat& image,
                                 int minX, int minY,
                                 int strideX, int strideY,
                                 double width, double height,
                                 const cv::Mat& mask,
                                 cv::Mat& scores,
                                 PointScores* locations);

  // Disallow evil constructor
  VisualUtilityEstimator(const VisualUtilityEstimator&);
  void operator=(const VisualUtilityEstimator&);

  // Friend the subclasses that encapsulate other visual utility
  // estimators so that they can call the protected functions.
  friend class ScaledDetectorWrapper;

};

// Estimates motion as the LAB different between two frames that have
// been aligned.
class LABMotionVUEstimator : public VisualUtilityEstimator {
public:
  LABMotionVUEstimator(const TransformEstimator& transformEstimator,
                       double paretoThreshold,
                       double distDecay,
                       int openingSize)
    : VisualUtilityEstimator(), lastTransform_(),
      transformEstimator_(transformEstimator),
      paretoThreshold_(paretoThreshold),
      distDecay_(distDecay),
      openingSize_(openingSize),
      lastGreyImage_(), lastLabImage_(){
    // For backwards compatibility
    if (distDecay > 1.0) { distDecay_ = distDecay / 10; }
  }
  virtual ~LABMotionVUEstimator();

  virtual
  cv::Mat_<double> CalculateVisualUtility(const cv::Mat& image,
                                          double time);

  virtual const cv::Mat* GetLastTransform() const;

private:
  cv::Mat lastTransform_;

  const TransformEstimator& transformEstimator_;

  // Fraction of pixels to grab assuming a pareto distribution
  double paretoThreshold_;
  // How quickly the threshold decays in the sigmoid function
  double distDecay_;
  // The size of the filter for the opening operation. Should be an odd number
  int openingSize_;

  // The last image input into the model both in LAB space and in greyscale
  cv::Mat_<double> lastGreyImage_;
  cv::Mat_<cv::Vec3d> lastLabImage_;

  // Calculates the value to use as a threshold on dist assuming a
  // preto distribution.
  double CalculateParetoThreshold(const cv::Mat_<double>& dist);

};

// Visual Utility Estimator that calculates spectral residual saliency
// for each frame. This is from Hou X. et al. Saliency Detection: A
// Spectral Residual Approach
class SpectralSaliency :  public VisualUtilityEstimator {
public:
  SpectralSaliency() : VisualUtilityEstimator() {}

  virtual ~SpectralSaliency();

  // Workhorse function that calculates the visual utility for every frame
  virtual
  cv::Mat_<double> CalculateVisualUtility(const cv::Mat& image,
                                          double time);

};

// This is for comparison and sets the bounding boxes identified as an
// object to 1.0 and everything else to 0.0.
//
// Needs the ground truth file as input which has for each line, the
// information about one object in the form:
// <time> <frame Number> <target Id> <annotation type> <x1> <y1> <x2> <y2> ...
//
// We key off of the time.
class HimaGroundTruth : public VisualUtilityEstimator {

public:
  HimaGroundTruth(const std::string& groundTruthFile)
    : VisualUtilityEstimator() {}

  virtual ~HimaGroundTruth();

  virtual
  cv::Mat_<double> CalculateVisualUtility(const cv::Mat& image,
                                          double time);

private:

  void ParseFile(const std::string& filename);
};

// A wrapper version of visual utility that uses the relative entropy
// of a region relative to the average in a frame in order to
// calculate the visual utility in a bounding box.
class RelativeEntropyVUWrapper : public VisualUtilityEstimator {
public:

  // Takes ownership of the underlying visual utility estimator
  RelativeEntropyVUWrapper(VisualUtilityEstimator* baseEstimator)
    : baseEstimator_(baseEstimator) {ROS_ASSERT(baseEstimator);}

  virtual ~RelativeEntropyVUWrapper();

  // Workhorse function that calculates the visual utility for every frame
  virtual
  cv::Mat_<double> CalculateVisualUtility(const cv::Mat& image,
                                          double time) {
    return baseEstimator_->CalculateVisualUtility(image, time);
  }  

  // Some of the visual utility estimators can calculate a transform.
  // If they do, this function will return the transform from the
  // frame 2 back to the last frame that we calculated the visual
  // utility for.
  //
  // If there's no transform that is found, NULL is returned.
  virtual const cv::Mat* GetLastTransform() const {
    return baseEstimator_->GetLastTransform();
  }

protected:
  virtual bool InitBoxCalculator(const cv::Mat& image, double time);

  virtual double CalculateVisualUtilityOfBox(int x, int y, int h,
                                             int w);

private:
  boost::scoped_ptr<VisualUtilityEstimator> baseEstimator_;

  // The integral image of the base visual utility
  boost::scoped_ptr<cv::Mat_<double> > integralVu_;
};

// A visual utility estimator that uses the laplacian image.
class LaplacianVU : public VisualUtilityEstimator {
public:
  LaplacianVU(int ksize) : ksize_(ksize){}
  virtual ~LaplacianVU();

  virtual
  cv::Mat_<double> CalculateVisualUtility(const cv::Mat& image,
                                          double time);
private:
  // Aperature size for the laplacian. Must be odd and positive
  int ksize_;
};

// A visual utility estimator that uses the distance between the
// histogram of the area in the center and that from the surround. The
// maximum from multiple possible scales is taken. This is only defined
// for regions because the size of the center surround is relative to
// the size of the bounding box in question.
class CenterSurroundHistogram : public VisualUtilityEstimator {
public:
  // A scale of 1.0 is equal to the size of the query region for the
  // center, while the surround is of equal area.
  // Inputs:
  // surroundScales - list of surroundScales to use
  // distType - Type of distance to calculate. Can be one of "chisq",
  //            "correl", "emd", "intersect", "bhattacharyya"
  CenterSurroundHistogram(const std::vector<double>& surroundScales,
                          const std::string& distType);
  virtual ~CenterSurroundHistogram();

  virtual
  cv::Mat_<double> CalculateVisualUtility(const cv::Mat& image,
                                          double time);

protected:
  virtual bool InitBoxCalculator(const cv::Mat& image, double time);
  
  virtual double CalculateVisualUtilityOfBox(int x, int y, int h, int w);

private:
  std::vector<double> surroundScales_;
  int distType_;

  boost::shared_ptr<cv_utils::IntegralHistogram<int> > integralHist_;
};

// Wrapper for the objectness measure in the objectness node
class Objectness : public VisualUtilityEstimator {
public:
  Objectness();
  virtual ~Objectness();

  virtual void CalculateVisualUtility(const cv::Mat& image,
                                      const std::vector<cv::Rect>& rois,
                                      double time,
                                      std::vector<ROIScore>* vuOut);

  virtual cv::Mat CalculateVisualUtility(const cv::Mat& image,
                                         int minX, int minY,
                                         int minW, int minH,
                                         int strideX, int strideY,
                                         double strideW, double strideH,
                                         bool fixAspect,
                                         double time,
                                         const cv::Mat& mask=cv::Mat());

  // Not implemented and will just throw an error
  virtual
  cv::Mat_<double> CalculateVisualUtility(const cv::Mat& image,
                                          double time);
  

private:
  objectness::Objectness impl_;
};

// A visual utility estimator composed of precomputed results stored
// in a ros bag.
class ROSBag : public VisualUtilityEstimator {
public:
  ROSBag(const std::string& filename);
  virtual ~ROSBag();

  virtual void CalculateVisualUtility(const std::string& filename,
                                      const std::vector<cv::Rect>& rois,
                                      double time,
                                      std::vector<ROIScore>* vuOut);
  virtual void CalculateVisualUtility(const cv::Mat& image,
                                      const std::vector<cv::Rect>& rois,
                                      double time,
                                      std::vector<ROIScore>* vuOut);
  virtual
  cv::Mat_<double> CalculateVisualUtility(const cv::Mat& image,
                                          double time);

private:
  struct ResultKey {
    ResultKey(const std::string& _filename, const cv::Rect _rect);
    bool operator==(const ResultKey& b) const {
      return filenameHash == b.filenameHash && rect == b.rect;
    }

    std::size_t filenameHash;
    cv::Rect rect;
  };
  struct ResultKeyHash {
    std::size_t operator()(const ResultKey& s) const;
  };

  boost::scoped_ptr<VisualUtilityEstimator> baseEstimator_;
  boost::scoped_ptr<TransformEstimator> transformEstimator_;
  hash_map<ResultKey, float, ResultKeyHash> lut_; 

  // Uses the parameters from the bag to create the baseEstimator_
  void CreateBaseEstimator(const rosbag::Bag& bag);

  // Reads the results topic and loads the computed values that are
  // encapsulated in the bag.
  void LoadBagResults(const rosbag::Bag& bag);

  // Evil constructors
  ROSBag(const ROSBag&);
  void operator=(const ROSBag&);
};

class HOGDetector : public VisualUtilityEstimator {
public:
  HOGDetector(const std::string& modelFile,
              bool useDefaultPeopleDetector,
              cv::Size winStride=cv::Size(),
              bool doCache=true);
  virtual ~HOGDetector();

  virtual void CalculateVisualUtility(const cv::Mat& image,
                                      const std::vector<cv::Rect>& rois,
                                      double time,
                                      std::vector<ROIScore>* vuOut);
  virtual
  cv::Mat_<double> CalculateVisualUtility(const cv::Mat& image,
                                          double time);

private:
  hog_detector::HogDetectorImpl impl_;

  // Evil constructors
  HOGDetector(const HOGDetector&);
  void operator=(const HOGDetector&);
};

// A HOG detector built using integral histograms.
class IntegralHOGDetector : public VisualUtilityEstimator {
public:
  IntegralHOGDetector(const std::string& modelFile,
                      const cv::Size& winStride);
  virtual ~IntegralHOGDetector();

protected:
  virtual bool InitBoxCalculator(const cv::Mat& image, double time);

  virtual double CalculateVisualUtilityOfBox(int x, int y, int h,
                                             int w);

  virtual
  cv::Mat_<double> CalculateVisualUtility(const cv::Mat& image,
                                          double time);

private:
  hog_detector::IntegralHogDetector impl_;
  boost::scoped_ptr<cv_utils::IntegralHistogram<float> > hist_;
  cv::Mat_<float> histSum_;

  // Evil constructors
  IntegralHOGDetector(const IntegralHOGDetector&);
  void operator=(const IntegralHOGDetector&);
};

// A Cascaded HOG detector built on integral histograms/
class IntegralHOGCascade : public VisualUtilityEstimator {
public:
  IntegralHOGCascade(const std::string& modelFile,
                     const cv::Size& winStride);
  virtual ~IntegralHOGCascade();

protected:
  virtual bool InitBoxCalculator(const cv::Mat& image, double time);

  virtual double CalculateVisualUtilityOfBox(int x, int y, int h,
                                             int w);

  virtual
  cv::Mat_<double> CalculateVisualUtility(const cv::Mat& image,
                                          double time);

private:
  hog_detector::IntegralHogCascade impl_;
  boost::scoped_ptr<cv_utils::IntegralHistogram<float> > hist_;
  cv::Mat_<float> histSum_;

  // Evil constructors
  IntegralHOGCascade(const IntegralHOGCascade&);
  void operator=(const IntegralHOGCascade&);
};


// A viola-jones like detector. See the OpenCV CascadeClassifier for
// more details. Note that the timing of this detector is a little
// weird for two reasons. First, the OpenCV 2+ implementation requires
// resizing the image, which is slow, so we need to not could the
// image. Second, the runtime is dependent on the number of stages in
// the classifier, so we have to estimate the timing of the
// computation after n stages of the classifier.
class CascadeDetector : public VisualUtilityEstimator {
public:
  CascadeDetector(const std::string& modelFile);
  virtual ~CascadeDetector();

  virtual void CalculateVisualUtility(const cv::Mat& image,
                                      const std::vector<cv::Rect>& rois,
                                      double time,
                                      std::vector<ROIScore>* vuOut);
  virtual
  cv::Mat_<double> CalculateVisualUtility(const cv::Mat& image,
                                          double time);

private:
  cascade_detector::DetailedCascadeDetector impl_;;

  // Evil constructors
  CascadeDetector(const CascadeDetector&);
  void operator=(const CascadeDetector&);
};

// A wrapper detector that computes another detector, but on a scaled image.
class ScaledDetectorWrapper : public VisualUtilityEstimator {
public:
  // Takes ownership of the underlying visual utility estimator
  // Scale factor is how much to scale the image (e.g. 2.0 will double the size of each dimension
  ScaledDetectorWrapper(VisualUtilityEstimator* baseEstimator,
                        double scaleFactor)
    : baseEstimator_(baseEstimator), scaleFactor_(scaleFactor) {
    ROS_ASSERT(baseEstimator);
  }

  virtual ~ScaledDetectorWrapper();


  virtual
  cv::Mat_<double> CalculateVisualUtility(const cv::Mat& image,
                                          double time);

  virtual
  void CalculateVisualUtility(const cv::Mat& image,
                              const std::vector<cv::Rect>& rois,
                              double time,
                              std::vector<ROIScore>* vuOut);
  

  virtual const cv::Mat* GetLastTransform() const {
    return baseEstimator_->GetLastTransform();
  }

  virtual double* GetLastRuntime() const {
    if (baseEstimator_->GetLastRuntime() != NULL) {
      lastRuntime_.reset(new double(*baseEstimator_->GetLastRuntime() + 
                                    resizeTime_.toSec()));
    }
    return lastRuntime_.get();
  }

protected:
  virtual bool InitBoxCalculator(const cv::Mat& image, double time);

  virtual double CalculateVisualUtilityOfBox(int x, int y, int h,
                                             int w);

private:
  boost::scoped_ptr<VisualUtilityEstimator> baseEstimator_;
  double scaleFactor_;
  ros::WallDuration resizeTime_;

  // Evil constructors
  ScaledDetectorWrapper(const ScaledDetectorWrapper&);
  void operator=(const ScaledDetectorWrapper&);
};



} // namespace

#endif  // __VISUAL_UTILITY_ESTIMATOR_H__
