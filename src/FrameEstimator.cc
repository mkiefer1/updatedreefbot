#include "visual_utility/FrameEstimator.h"

#include <ros/ros.h>
#include <limits>
#include <math.h>
#include <gsl/gsl_multimin.h>
#include "cv_blobs/BlobResult-Inline.h"
#include "cv_blobs/BlobFilters.h"
#include "cv_utils/DisplayImages.h"

using namespace cv;
using namespace std;
using namespace cv_blobs;
namespace visual_utility {

// -------- Useful functors for search through the mosaic --------

// Look for the maximal pixel value. Nothing fancy
struct MaxPixelFunc : public PixelValFunc {
  virtual double operator()(const double& val) { return val; };
};



// -------- end of functors -----------

FrameEstimator::~FrameEstimator() {}

FrameOfInterest FrameEstimator::ExpandFrame(const FrameOfInterest& frame) {
  FrameOfInterest retVal = frame;
  retVal.height *= frameExpansionFactor_;
  retVal.width *= frameExpansionFactor_;
  return retVal;
}

// ---------- MaxPointConstantFramesize -----------

MaxPointConstantFramesize::MaxPointConstantFramesize(
  const Size_<int>& framesize)
  : FrameEstimator(1.0), framesize_(framesize) {}

MaxPointConstantFramesize::~MaxPointConstantFramesize() {}

void MaxPointConstantFramesize::FindInterestingFrames(
  const VisualUtilityMosaic& mosaic,
  vector<FrameOfInterest>* frames) {
  ROS_ASSERT(frames != NULL);

  Point2f point;
  double maxVal = mosaic.MaximizeFunction(&MaxPixelFunc(), &point);

  if (!isfinite(maxVal)) {
    ROS_WARN_STREAM("The maximum value in the frame is not finite. It is "
                    << maxVal
                    << ". Returning the center of the frame");
    frames->push_back(FrameOfInterest(0, 0, framesize_.height,
                                      framesize_.width));
    return;
  }

  frames->push_back(FrameOfInterest(point.x, point.y, framesize_.height,
                                    framesize_.width));
}

// ----------------- RandomPointConstantFramesize -------------

RandomPointConstantFramesize::RandomPointConstantFramesize(
  const cv::Size_<int>& framesize)
  : FrameEstimator(1.0), framesize_(framesize), seed_(1695783),
    randomNumberGenerator_(), randomRange_(),
    randNum_(randomNumberGenerator_, randomRange_) {
  randomNumberGenerator_.seed(seed_);
}

RandomPointConstantFramesize::~RandomPointConstantFramesize() {}

void RandomPointConstantFramesize::FindInterestingFrames(
  const VisualUtilityMosaic& mosaic,
  vector<FrameOfInterest>* frames) {
  ROS_ASSERT(frames != NULL);
  
  // Pick a random location on the frame
  frames->push_back(FrameOfInterest((randNum_()-0.5)*fullFramesize_.width,
                                    (randNum_()-0.5)*fullFramesize_.height,
                                    framesize_.height,
                                    framesize_.width));
}

//  ---------------  DynamicResizeAroundMax --------------

DynamicResizeAroundMax::~DynamicResizeAroundMax() {}

// Function for maximizing the visual utility/area for a frame of a
// given center
struct OptimalFrameParams {
  OptimalFrameParams(const cv::Point2f& _frameCenter,
                     const VisualUtilityMosaic& _mosaic) 
    : frameCenter(_frameCenter), mosaic(_mosaic) {}
  const cv::Point2f& frameCenter;
  const VisualUtilityMosaic& mosaic;
};

// Now calculate the visual utility in the box centered at
// params->frameCenter, of width x[0] and height x[1]
double VuPerAreaInFrame(const gsl_vector* x, void* vParams) {
  OptimalFrameParams* params = reinterpret_cast<OptimalFrameParams*>(vParams);

  double width = gsl_vector_get(x, 0);
  double height = gsl_vector_get(x, 1);

  Point2f minCorner = params->frameCenter - Point2f(width/2.0, height/2.0);
  Point2f maxCorner = params->frameCenter + Point2f(width/2.0, height/2.0);

  double visualUtility = params->mosaic.GetIntegralValue(maxCorner) +
    params->mosaic.GetIntegralValue(minCorner) -
    params->mosaic.GetIntegralValue(Point2f(minCorner.x, maxCorner.y)) -
    params->mosaic.GetIntegralValue(Point2f(maxCorner.x, minCorner.y));
  return -(visualUtility / (width*height));
 
}

void DynamicResizeAroundMax::FindInterestingFrames(
  const VisualUtilityMosaic& mosaic,
  vector<FrameOfInterest>* frames) {

  ROS_ASSERT(frames != NULL);

  // Find the maximum value of visual utility
  Point2f maxPoint;
  double maxVal = mosaic.MaximizeFunction(&MaxPixelFunc(), &maxPoint);

  if (!isfinite(maxVal)) {
    ROS_WARN_STREAM("The maximum value in the frame is not finite. It is "
                    << maxVal
                    << ". Returning the center of the frame");
    frames->push_back(FrameOfInterest(0, 0, fullFramesize_.height,
                                      fullFramesize_.width));
    return;
  }

  // Now grow a frame around the maximum value to get the frame with
  // the best utility/area.
  gsl_multimin_function funcParams;
  OptimalFrameParams params(maxPoint, mosaic);
  funcParams.n = 2;
  funcParams.f = VuPerAreaInFrame;
  funcParams.params = reinterpret_cast<void*>(&params);

  // Set a starting point at the minimum frame size
  gsl_vector* x = gsl_vector_alloc(2);
  gsl_vector_set(x, 0, minFramesize_.width);
  gsl_vector_set(x, 1, minFramesize_.height);

  // Create the minimizer. We're going to use Nelson-Melder
  const gsl_multimin_fminimizer_type* minimizerT = 
    gsl_multimin_fminimizer_nmsimplex;
  gsl_multimin_fminimizer* minimizer =
    gsl_multimin_fminimizer_alloc(minimizerT, 2);
  
  gsl_vector* stepSize = gsl_vector_alloc(2);
  gsl_vector_set(stepSize, 0, 10.0);
  gsl_vector_set(stepSize, 1, 10.0);
  
  gsl_multimin_fminimizer_set(minimizer, &funcParams, x, stepSize);

  int status;
  int iter = 0;
  do {
    ++iter;

    status = gsl_multimin_fminimizer_iterate(minimizer);
    
    if (gsl_multimin_fminimizer_size(minimizer) < 3.0) {
      status = GSL_SUCCESS;
    }
  } while (status == GSL_CONTINUE && iter < 50);
  
  FrameOfInterest foi;

  if (status != GSL_SUCCESS) {
    ROS_WARN_STREAM("Search for best dynamic frame did not converge. "
                    "Returning the minimim frame.");
    foi = FrameOfInterest(maxPoint.x, maxPoint.y, minFramesize_.height,
                          minFramesize_.width);
  } else {
    ROS_INFO_STREAM("Converged after " << iter << " iterations.");
    foi = FrameOfInterest(maxPoint.x, maxPoint.y, gsl_vector_get(x, 1),
                          gsl_vector_get(x, 0));
  }

  gsl_vector_free(x);
  gsl_vector_free(stepSize);
  gsl_multimin_fminimizer_free(minimizer);

  frames->push_back(ExpandFrame(foi));
}

// ---------  HighRelativeEntropy  --------------

HighRelativeEntropy::~HighRelativeEntropy() {}

// A predicate that compares the relative entropy of a region. For example, to
// create an operator that returns true if the area is less than 500,
// use:
// AreaCompare<less<int> >(500)
template<typename ComparePred, typename T>
class EntropyCompare {
public:
  EntropyCompare(T thresh, const VisualUtilityMosaic& mosaic)
    : thresh_(thresh), mosaic_(mosaic) {}
  bool operator()(const Blob& blob) {
    double blobSum = 0;
    for (Blob::BlobContainer::const_iterator i = blob.begin();
         i != blob.end(); ++i) {
      blobSum += mosaic_.GetValue(*i);
    }
    
    double pB = blob.area() / mosaic_.GetSize();
    double rB = blobSum / mosaic_.GetSum();
    double entropy = 0;
    if (rB > pB) {
      entropy = rB * log(rB/pB) + (1-rB) * log((1-rB)/(1-pB));
    }
    return ComparePred()(entropy, thresh_);
  }

private:
  T thresh_;
  const VisualUtilityMosaic& mosaic_;

};

void HighRelativeEntropy::FindInterestingFrames(
  const VisualUtilityMosaic& mosaic,
  vector<FrameOfInterest>* frames) {
  ROS_ASSERT(frames != NULL);

  // Figure out the threshold for areas that are likely to be
  // interesting. This is just the value of the visual utility
  // assuming an uniform spatial distribution.
  double mSum = mosaic.GetSum();
  double mSize = mosaic.GetSize();
  double thresh = mSum / mSize;

  // Now find all the connected components above that threshold
  BlobResult<double> blobs;
  mosaic.FindConnectedComponents(thresh, &blobs);

  if (displayDebugImages_) {
    scoped_ptr<BlobResult<double> > visibleBlobs(blobs.copy());
    *visibleBlobs += Point2i(blobs.ImageSize().width/2.0, blobs.ImageSize().height/2.0);
    cv_utils::DisplayNormalizedImage(visibleBlobs->ToBinaryImage(),
                                     "Initial blobs");
  }

  // Next filter those blobs that are too small
  blobs.Filter(BoxAreaCompare<greater<int>, int>(minFrameArea_ / 
                                                 (frameExpansionFactor_ * 
                                                  frameExpansionFactor_)));

  if (displayDebugImages_ || keepDebugImage_) {
    scoped_ptr<BlobResult<double> > visibleBlobs(blobs.copy());
    *visibleBlobs += Point2i(blobs.ImageSize().width/2.0, blobs.ImageSize().height/2.0);
    if (displayDebugImages_) {
      cv_utils::DisplayNormalizedImage(visibleBlobs->ToBinaryImage(),
                                       "Only large blobs");
    }
    if (keepDebugImage_) {
      //lastDebugImage_ = visibleBlobs->ToBinaryImage();
    }
  }

  // Now filter out those blobs whose entropy is too small
  double minEntropy = log(mSize * (mSize + 1) / 2) / mSum;
  blobs.Filter(EntropyCompare<greater<double>, double>(minEntropy,
                                                       mosaic));

  if (displayDebugImages_ || keepDebugImage_) {
    scoped_ptr<BlobResult<double> > visibleBlobs(blobs.copy());
    *visibleBlobs += Point2i(blobs.ImageSize().width/2.0, blobs.ImageSize().height/2.0);
    if (displayDebugImages_) {
      cv_utils::DisplayNormalizedImage(visibleBlobs->ToBinaryImage(),
                                       "Blobs with high entropy");
    }
    if (keepDebugImage_) {
      lastDebugImage_ = visibleBlobs->ToBinaryImage();
    }
  }

  // Finally, for the blobs that are left, generate the frames of interest
  for (int i = 0; i < blobs.nBlobs(); ++i) {
    const Blob& blob = blobs.GetBlob(i);
    float width = blob.maxX() - blob.minX();
    float height = blob.maxY() - blob.minY();
    frames->push_back(ExpandFrame(FrameOfInterest(blob.minX() + width/2.0,
                                                  blob.minY() + height/2.0,
                                                  height,
                                                  width)));
  }

}

// --------- LocationListWithThreshold ---------

LocationListWithThreshold::~LocationListWithThreshold() {}

LocationListWithThreshold::LocationListWithThreshold(
  double thresh,
  const std::vector<cv::Rect>& rois,
  const cv::Size& framesize) 
  : FrameEstimator(1.0) {
  thresh_ = thresh;

  for (vector<Rect>::const_iterator i = rois.begin();
       i != rois.end(); ++i) {
    regions2Sample_.push_back(
      FrameOfInterest(i->x - (framesize.width - i->width)/2,
                      i->y - (framesize.height - i->height)/2,
                      i->height,
                      i->width));
  }
}

void LocationListWithThreshold::FindInterestingFrames(
  const VisualUtilityMosaic& mosaic,
  vector<FrameOfInterest>* frames) {
  ROS_ASSERT(frames);

  for (vector<FrameOfInterest>::const_iterator i = regions2Sample_.begin();
       i != regions2Sample_.end(); ++i) {
    Rect_<float> box(i->xCenter - i->width/2, i->yCenter - i->height/2,
                     i->width, i->height);

    // Calculates the relative entropy of the region compared to the
    // entire image.
    double pB = box.area() / mosaic.GetSize();
    double rB = mosaic.GetSumInBox(box) / mosaic.GetSum();
    double entropy = 0;
    if (rB > pB) {
      entropy = rB * log(rB/pB) + (1-rB) * log((1-rB)/(1-pB));
    } else {
      entropy = - pB * log(pB/rB) - (1-pB) * log((1-pB)/(1-rB));
    }

    if (entropy > thresh_) {
      frames->push_back(*i);
    }
  }
}

} // namespace
