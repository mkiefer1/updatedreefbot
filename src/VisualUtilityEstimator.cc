#include "visual_utility/VisualUtilityEstimator.h"

#include <ros/ros.h>
#include <rosbag/view.h>
#include <limits>
#include <math.h>
#include <boost/scoped_ptr.hpp>
#include <boost/foreach.hpp>
#include <boost/functional/hash.hpp>
#include <boost/unordered_map.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include "visual_utility/cvutils-inl.h"
#include "cv_utils/DisplayImages.h"
#include "cv_utils/IntegralHistogram-Inl.h"
#include "visual_utility/Parameter.h"
#include "visual_utility/VisualUtilityEstimation.h"
#include "base/StringHash.h"
#include "visual_utility/VisualUtilityROSParams.h"
#include "hog_detector/integral_hog_detector_inl.h"

using namespace cv;
using namespace boost;
using namespace std;

namespace cv {
inline std::size_t hash_value(const cv::Rect& key) {
  size_t seed = 7823;
  hash_combine(seed, key.x);
  hash_combine(seed, key.y);
  hash_combine(seed, key.height);
  hash_combine(seed, key.width);
  return seed;
}

inline std::size_t hash_value(const cv::Size& key) {
  size_t seed = 54326;
  hash_combine(seed, key.height);
  hash_combine(seed, key.width);
  return seed;
}
} // end namespace

namespace visual_utility {

// ---------- Start VisualUtilityEstimator -----------

VisualUtilityEstimator::~VisualUtilityEstimator() {}

const Mat* VisualUtilityEstimator::GetLastTransform() const {
  return NULL;
}

void VisualUtilityEstimator::CalculateVisualUtility(
  const string& filename,
  const vector<Rect>& rois,
  double time,
  std::vector<ROIScore>* vuOut) {
  // Open up the file
  Mat cvImage = imread(filename);

  // Calculate the visual utility
  CalculateVisualUtility(cvImage, rois, time, vuOut);
}

void VisualUtilityEstimator::CalculateVisualUtility(
  const cv::Mat& image,
  const vector<Rect>& rois,
  double time,
  std::vector<ROIScore>* vuOut) {

  ROS_ASSERT(vuOut);
  lastRuntime_.reset(NULL);

  // Start by initializing anything that's needed with the image
  ros::WallTime startTime = ros::WallTime::now();
  InitBoxCalculator(image, time);
  ros::WallDuration initTime = ros::WallTime::now() - startTime;

  ROS_INFO_STREAM("Processing time for initialization: " << initTime.toSec());

  // Now collect all the rois of interest for each box size
  unordered_map<Size, shared_ptr<PointScores> > pointLists;
  vuOut->resize(rois.size());
  int i = 0;
  for(vector<Rect>::const_iterator regionI = rois.begin();
      regionI != rois.end(); ++regionI, ++i) {
    Size winSize(regionI->width, regionI->height);
    unordered_map<Size, shared_ptr<PointScores> >::iterator pointsPtr =
      pointLists.find(winSize);
    if (pointsPtr == pointLists.end()) {
      pointsPtr = pointLists.insert(pair<Size, shared_ptr<PointScores> >(
        winSize,
        shared_ptr<PointScores>(new PointScores()))).first;
    }
    (*vuOut)[i].second = *regionI;
    pointsPtr->second->push_back(pair<double*, Point>(&((*vuOut)[i].first),
                                                      Point(regionI->x,
                                                            regionI->y)));
  }

  // Now for each box size, calculate the visual utility scores across
  // the image.
  startTime = ros::WallTime::now();
  for(unordered_map<Size, shared_ptr<PointScores> >::const_iterator
        pointsI = pointLists.begin();
      pointsI != pointLists.end(); ++pointsI) {
    CalculateVisualUtilityBoxAcrossImage(pointsI->first.width,
                                         pointsI->first.height,
                                         *pointsI->second);
  }

  ros::WallDuration calculationTime = ros::WallTime::now() - startTime;
  ROS_INFO_STREAM("Processing time for calculation: " << calculationTime);
  if (lastRuntime_.get() == NULL) {
    lastRuntime_.reset(new double((initTime + calculationTime).toSec()));
  }
}

// Helper function for the following CalculateVisualUtility that
// creates a list of points to evaluate at based on a grid and a mask.
void VisualUtilityEstimator::GetLocationsForEvaluation(
  const Mat& image,
  int minX, int minY,
  int strideX, int strideY,
  double width, double height,
  const Mat& mask,
  Mat& scores,
  PointScores* locations) {
  ROS_ASSERT(mask.empty() || mask.type() == CV_8U);
  ROS_ASSERT(scores.type() == CV_64F);
  ROS_ASSERT(locations);
  locations->clear();

  for (int i = 0, curX = minX; curX < image.cols-width; ++i) {
    for (int j = 0, curY = minY; curY < image.rows-height; ++j) {
      if (mask.empty() || mask.at<uint8_t>(i,j)) {
        locations->push_back(pair<double*, Point>(scores.ptr<double>(i,j),
                                                  Point(curX,curY)));
      }
      curY += strideY;
    }
    curX += strideX;
  }
}

void VisualUtilityEstimator::GetGridHeightsAndWidths(const Mat& image,
                                                     int minX, int minY,
                                                     int minW, int minH,
                                                     double strideW,
                                                     double strideH,
                                                     bool fixAspect,
                                                     vector<double>* widths,
                                                     vector<double>* heights){
  ROS_ASSERT(widths);
  ROS_ASSERT(heights);

  if (fixAspect) {
    strideH = strideW;
  }

  for (double curW = minW; curW < image.cols-minX; curW *= strideW) {
    widths->push_back(cvRound(curW));
  }

  for (double curH = minH; curH < image.rows-minY; curH *= strideH) {
    heights->push_back(cvRound(curH));
  }

  // If the aspect ratio is fixed, truncate the list of sizes to the
  // smaller one.
  if (fixAspect) {
    if (heights->size() > widths->size()) {
      heights->resize(widths->size());
    } else {
      widths->resize(heights->size());
    }
  }
  
}

void VisualUtilityEstimator::InitializeScoreGrid(
  const Mat& image,
  int minX, int minY,
  int strideX, int strideY,
  bool fixAspect,
  const vector<double>& widths,
  const vector<double>& heights,
  Mat* scoreGrid) {
  ROS_ASSERT(scoreGrid);

  vector<int> scoreSize;
  scoreSize.push_back(widths.size());
  if (!fixAspect) scoreSize.push_back(heights.size());
  scoreSize.push_back(
    std::max<int>(1, (image.cols - minX - widths[0]) / strideX));
  scoreSize.push_back(
    std::max<int>(1, (image.rows - minY - heights[0]) / strideY));
  *scoreGrid = cv::Mat(scoreSize.size(), &scoreSize[0], CV_64F, 
                   -numeric_limits<double>::infinity());
}

cv::Mat VisualUtilityEstimator::CalculateVisualUtility(
  const cv::Mat& image,
  int minX, int minY,
  int minW, int minH,
  int strideX, int strideY,
  double strideW, double strideH,
  bool fixAspect,
  double time,
  const cv::Mat& mask) {
  Mat scores;

  lastRuntime_.reset(NULL);

  // Start by initializing anything that's needed with the image
  ros::WallTime startTime = ros::WallTime::now();
  InitBoxCalculator(image, time);
  ros::WallDuration initTime = ros::WallTime::now() - startTime;

  vector<double> widths;
  vector<double> heights;
  GetGridHeightsAndWidths(image, minX, minY, minW, minH, strideW, strideH,
                          fixAspect, &widths, &heights);

  // Build the output score matrix
  InitializeScoreGrid(image, minX, minY, strideX, strideY, fixAspect,
                      widths, heights,
                      &scores);
  ROS_ASSERT(mask.empty() ||
             (mask.dims == scores.dims &&
              mask.size == scores.size));

  
  // Now walk through the different size of boxes
  PointScores locations;
  ros::WallDuration calculationTime(0);
  for (unsigned int widthI = 0u; widthI < widths.size(); ++widthI) {
    if (fixAspect) {
      Mat scorePlane(2, &scores.size[1], CV_64F, 
                     scores.ptr(widthI), &scores.step[1]);
      GetLocationsForEvaluation(image,
                                minX, minY, strideX, strideY,
                                widths[widthI],
                                heights[widthI],
                                (mask.empty() ? mask :
                                 Mat(2, &mask.size[1], mask.type(),
                                     const_cast<uchar*>(mask.ptr(widthI)),
                                     &mask.step[1])),
                                scorePlane,
                                &locations);

      startTime = ros::WallTime::now();
      CalculateVisualUtilityBoxAcrossImage(widths[widthI],
                                           heights[widthI],
                                           locations);
      calculationTime += ros::WallTime::now() - startTime;
      
    } else {
      for (unsigned int heightI = 0u; heightI < heights.size(); ++heightI) {
        Mat scorePlane(2, &scores.size[2], CV_64F, 
                       scores.ptr(widthI, heightI),
                       &scores.step[2]);
        GetLocationsForEvaluation(image, 
                                  minX, minY, strideX, strideY,
                                  widths[widthI],
                                  heights[heightI],
                                  (mask.empty() ? mask :
                                   Mat(2, &mask.size[2], mask.type(),
                                       const_cast<uchar*>(mask.ptr(widthI,
                                                                  heightI)),
                                       &mask.step[2])),
                                  scorePlane,
                                  &locations);
        
        startTime = ros::WallTime::now();
        CalculateVisualUtilityBoxAcrossImage(widths[widthI],
                                             heights[heightI],
                                             locations);
        calculationTime += ros::WallTime::now() - startTime;
      }
    }
  }

  if (lastRuntime_.get() == NULL) {
    lastRuntime_.reset(new double((initTime + calculationTime).toSec()));
  }

  return scores;
}

bool VisualUtilityEstimator::InitBoxCalculator(const cv::Mat& image,
                                               double time) {
  ROS_FATAL("InitBoxCalculator is not implemented");
  ROS_ASSERT(false);
  return false;
}

void VisualUtilityEstimator::CalculateVisualUtilityBoxAcrossImage(
    double width, double height,
    const PointScores& locations) {

  for (PointScores::const_iterator pointI = locations.begin();
       pointI != locations.end();
       ++pointI) {
    *pointI->first = CalculateVisualUtilityOfBox(pointI->second.x,
                                                 pointI->second.y,
                                                 cvRound(height),
                                                 cvRound(width));
  }
  
}


double VisualUtilityEstimator::CalculateVisualUtilityOfBox(int x, int y,
                                                           int h, int w) {
  ROS_FATAL("Calculating the visual utility of a box is not implemented. "
            "Try wrapping it in one of the wrapper visual "
            "utility estimators like RelativeEntropyVUWrapper");
  ROS_ASSERT(false);
  return 0; // To make compiler happy
}

//-------- Start LABMotionVUEstimator --------------

LABMotionVUEstimator::~LABMotionVUEstimator() {}

const Mat* LABMotionVUEstimator::GetLastTransform() const {
  if (lastTransform_.empty()) {
    return NULL;
  }
  return &lastTransform_;
}

//#define GREY_OVERRIDE 0

cv::Mat_<double> LABMotionVUEstimator::CalculateVisualUtility(
  const Mat& image, double time) {
  
  if (image.channels() != 3) {
    ROS_ERROR("Image is not in color. We need a BGR image");
    return cv::Mat_<double>();
  }

  if (image.depth() != CV_8U) {
    ROS_FATAL("For speed reasons, the image must be still in its natural "
              "8-bit format");
    return cv::Mat_<double>();
  }

  // Get the LAB and greyscale images
  Mat_<uchar> tmpGreyImage;
  Mat_<double> curGreyImage;
  cvtColor(image, tmpGreyImage, CV_BGR2GRAY);
  tmpGreyImage.convertTo(curGreyImage, CV_64FC1, 1.0/255);

  // The L value in the image is scaled to be 0-255 instead of 0-100,
  // so we need to rescale it
  Mat_<Vec3b> tmpLabImage;
  Mat_<Vec3d> curLabImage;
  cvtColor(image, tmpLabImage, CV_BGR2Lab);
  tmpLabImage.assignTo(curLabImage, CV_64FC3);
  vector<Mat> channels;
  split(curLabImage, channels);
  channels[0] *= 100./255.;
  merge(channels, curLabImage);

  if (lastLabImage_.empty()) {
    // This is the first image added
    lastLabImage_ = curLabImage;
    lastGreyImage_ = curGreyImage;
    return cv::Mat_<double>();
  }

  // Calculate the affine transform to convert the last image to the
  // current one
  Mat lastTransform_ = transformEstimator_.EstimateTransform(lastGreyImage_,
                                                             curGreyImage);

  // Now warp the old image to the current one
#ifdef GREY_OVERRIDE
  Mat_<double> warpedImage;
  if (!lastTransform_.empty()) {
    warpedImage =  transformEstimator_.ApplyTransform(
      lastGreyImage_,
      lastTransform_,
      geo::BORDER_TRANSPARENT,
      0.0,
      curGreyImage);
  } else {
    warpedImage = lastGreyImage_;
  }
#else
  Mat_<Vec3d> warpedImage ;
  if (!lastTransform_.empty()) {
    warpedImage = transformEstimator_.ApplyTransform(
      lastLabImage_,
      lastTransform_,
      geo::BORDER_TRANSPARENT,
      Vec3d(0,0,0),
      curLabImage);
  } else {
    warpedImage = lastLabImage_;
  }
#endif

  // Next calculate the distance between the two images
  Mat_<float> dist(image.rows, image.cols, 0.0);

#ifdef GREY_OVERRIDE
  dist = abs(curGreyImage - warpedImage);
#else
  Mat_<Vec3d> diff = curLabImage - warpedImage;
  diff = diff.mul(diff);

  vector<Mat> diffChannels;
  split(diff, diffChannels);
  for (unsigned int i = 0u; i < diffChannels.size(); i++) {
    add(dist, diffChannels[i], dist, noArray(), CV_32F);
    //dist += diffChannels[i];
  }
  sqrt(dist, dist);
#endif

  lastLabImage_ = curLabImage;
  lastGreyImage_ = curGreyImage;

  // Do erode and dilate (aka openeing)
  morphologyEx(dist, dist, MORPH_OPEN,
               Mat_<int>::ones(openingSize_, openingSize_));

  return dist;
}

// Functor for calculating the sums needed for the alpha calculation
struct SumForAlphaFunctor {
  SumForAlphaFunctor(int& _nValidPixels,
                     double& _paretoSum,
                     double _logMode)
    : nValidPixels(_nValidPixels), paretoSum(_paretoSum), logMode(_logMode){}

  inline void operator()(const double val) {
    const double lVal = log(val);
    if (lVal >= logMode) {
      nValidPixels++;
      paretoSum += lVal - logMode;
    }
  }

  int& nValidPixels;
  double& paretoSum;
  const double logMode;
  
};

// Functor for building up the histogram
struct FillHistogramFunctor {
  FillHistogramFunctor(vector<double>& hist, double minVal, double maxVal)
    : hist_(hist), minVal_(minVal), maxVal_(maxVal),
      histFactor((hist.size()-1)/(maxVal-minVal)) {}

  inline void operator()(double val) {
    if (val >= minVal_ && val <= maxVal_) {
      hist_[static_cast<int>(histFactor*(val-minVal_))] += val;
    }
  }
    
    
  vector<double>& hist_;
  const double minVal_;
  const double maxVal_;
  const double histFactor;
};

double LABMotionVUEstimator::CalculateParetoThreshold(
  const Mat_<double>& dist) {
  // Fist find the range of distance values
  double maxVal;
  double minVal;
  minMaxLoc(dist, &minVal, &maxVal);

  if (minVal == maxVal) {
    return std::numeric_limits<double>::infinity();
  }

  // Calculate the histogram
  const int HIST_SIZE = 256;
  vector<double> hist(HIST_SIZE+1, 0.0);
  double histFactor = ((double)HIST_SIZE)/(maxVal - minVal);
  double bucketSize = 1./histFactor;
  cvutils::ApplyToEachElement<FillHistogramFunctor, double>(dist,
    FillHistogramFunctor(hist, minVal, maxVal));

  // Calculate the mode since we'll ignore points below that
  int modeBucket=0;
  double modeVal = 0;
  for (unsigned int i = 0u; i < hist.size(); ++i) {
    if (hist[i] > modeVal) {
      modeVal = hist[i];
      modeBucket = i;
    }
  }
  double mode = ((double)modeBucket)/histFactor + minVal + bucketSize/2;
  double logMode = log(mode);

  // Now calculate the alpha parameter
  int nValidPixels = 0;
  double paretoSum = 0;
  cvutils::ApplyToEachElement<SumForAlphaFunctor, double>(dist,
    SumForAlphaFunctor(nValidPixels, paretoSum, logMode));
  double alpha = nValidPixels / paretoSum;

  // Alternate way to calcuate alpha using the buckets
  /*int nValidPixels2 = 0;
  double paretoSum2 = 0;
  for (int i = 0; i < HIST_SIZE; ++i) {
    const double bucketCenter = ((double)i)/histFactor + minVal +
      bucketSize/2;
    if (bucketCenter >= mode) {
      nValidPixels2 += hist[i];
      paretoSum2 += hist[i]*(log(bucketCenter) - logMode);
    }
  }
  double alpha2 = nValidPixels2 / paretoSum2;

  ROS_WARN_STREAM("Alpha1: " << alpha << " Alpha2: " << alpha2);*/

  // Finally calculate the threshold to where we have paretoThresh_
  // fraction of the pixels
  return exp((alpha*logMode-log(paretoThreshold_)) / alpha);
}

// ------------ Start SpectralSaliency ------------
SpectralSaliency::~SpectralSaliency() {}

Mat_<double> SpectralSaliency::CalculateVisualUtility(
  const Mat& image, double time) {
  Mat_<double> retval;

  // Get a floating point grey version of the image
  /*Mat tmpGreyImage;
  if (image.channels() != 1) {
    cvtColor(image, tmpGreyImage, CV_BGR2GRAY);
  } else {
    tmpGreyImage = image;
  }
  Mat_<double> greyImage;
  tmpGreyImage.convertTo(greyImage, CV_64FC1, 1.0/255);
  Mat_<Vec<double, 2> > greyComplex;
  vector<Mat> greyGroup;
  greyGroup.push_back(greyImage);
  greyGroup.push_back(Mat_<double>::zeros(image.rows, image.cols));
  merge(greyGroup, greyComplex);

  Mat_<Vec<double, 2> > fftMat(image.rows, image.cols);
  dft(greyComplex, fftMat, DFT_COMPLEX_OUTPUT);

  vector<Mat> splitFftMat;
  split(fftMat, splitFftMat);

  Mat_<double> phaseMat;
  Mat_<double> logAmplitude;
  cartToPolar(splitFftMat[0], splitFftMat[1], logAmplitude, phaseMat);
  log(logAmplitude, logAmplitude);

  Mat_<double> spectralResidual;
  boxFilter(logAmplitude, spectralResidual, spectralResidual.depth(),
            Size(3, 3));
  spectralResidual = logAmplitude - spectralResidual;

  polarToCart(spectralResidual, phaseMat, splitFftMat[0], splitFftMat[1]);
  merge(splitFftMat, fftMat);

  Mat_<Vec<double, 2> > complexSaliency;
  dft(fftMat, complexSaliency, DFT_COMPLEX_OUTPUT | DFT_INVERSE);
  vector<Mat> splitSaliency;
  split(complexSaliency, splitSaliency);

  retval = splitSaliency[0].mul(splitSaliency[0]) +
    splitSaliency[1].mul(splitSaliency[1]);*/

  // The 32-bit floating point version to speed things up
  Mat tmpGreyImage;
  if (image.channels() != 1) {
    cvtColor(image, tmpGreyImage, CV_BGR2GRAY);
  } else {
    tmpGreyImage = image;
  }
  Mat_<float> greyImage;
  tmpGreyImage.convertTo(greyImage, CV_32FC1, 1.0/255);
  Mat_<Vec<float, 2> > greyComplex;
  vector<Mat> greyGroup;
  greyGroup.push_back(greyImage);
  greyGroup.push_back(Mat_<float>::zeros(image.rows, image.cols));
  merge(greyGroup, greyComplex);

  Mat_<Vec<float, 2> > fftMat(image.rows, image.cols);
  dft(greyComplex, fftMat, DFT_COMPLEX_OUTPUT);

  vector<Mat> splitFftMat;
  split(fftMat, splitFftMat);

  Mat_<float> phaseMat;
  Mat_<float> logAmplitude;
  cartToPolar(splitFftMat[0], splitFftMat[1], logAmplitude, phaseMat);
  log(logAmplitude, logAmplitude);

  Mat_<float> spectralResidual;
  boxFilter(logAmplitude, spectralResidual, spectralResidual.depth(),
            Size(3, 3));
  spectralResidual = logAmplitude - spectralResidual;

  polarToCart(spectralResidual, phaseMat, splitFftMat[0], splitFftMat[1]);
  merge(splitFftMat, fftMat);

  Mat_<Vec<float, 2> > complexSaliency;
  dft(fftMat, complexSaliency, DFT_COMPLEX_OUTPUT | DFT_INVERSE);
  vector<Mat> splitSaliency;
  split(complexSaliency, splitSaliency);

  retval = splitSaliency[0].mul(splitSaliency[0]) +
    splitSaliency[1].mul(splitSaliency[1]);
  
  return retval;
}

// ------- Start RelativeEntropyVUWrapper

RelativeEntropyVUWrapper::~RelativeEntropyVUWrapper() {}

bool RelativeEntropyVUWrapper::InitBoxCalculator(const cv::Mat& image,
                                                 double time) {
  // First get the visual utility of the entire image
  Mat_<double> baseVu = baseEstimator_->CalculateVisualUtility(image, time);
  
  // Now get the integral image so we can compute more quickly
  integralVu_.reset(new Mat_<double>());
  integral(baseVu, *integralVu_, CV_64F);

  
  return true;
}

double RelativeEntropyVUWrapper::CalculateVisualUtilityOfBox(int x, int y,
                                                             int h, int w) {
  const Mat_<double>& intImage = *integralVu_;

  double imArea = (intImage.rows-1)*(intImage.cols-1);
  double imSum = intImage[intImage.rows-1][intImage.cols-1];
  double pB = w*h / imArea;
  double rB = intImage[y][x] + intImage[y+h][x+w] -
    intImage[y][x+w] - intImage[y+h][x];
  double entropy = 0;
  if (imSum < 1e-10 || rB < 1e-10 || pB < 1e-10) {
    // We basically have a zero value, so force the entropy to be zero
  } else {
    rB /= imSum;
    if (rB > pB) {
      entropy = rB * log(rB/pB) + (1-rB) * log((1-rB)/(1-pB));
    } else {
      entropy = - pB * log(pB/rB) - (1-pB) * log((1-pB)/(1-rB));
    }
  }

  if (isnan(entropy)) {
    ROS_WARN_STREAM("Somehow we have the entropy being a NaN."
                    << " pB = " << pB
                    << " rB = " << rB
                    << " imSum = " << imSum);
  }

  return entropy;
}

// --------- Start LaplacianVU -------------
LaplacianVU::~LaplacianVU() {}

Mat_<double> LaplacianVU::CalculateVisualUtility(const cv::Mat& image,
                                                 double time) {
  Mat_<double> retval;

  Mat tmpGreyImage;
  if (image.channels() != 1) {
    cvtColor(image, tmpGreyImage, CV_BGR2GRAY);
  } else {
    tmpGreyImage = image;
  }
  Mat_<double> greyImage;
  tmpGreyImage.convertTo(greyImage, CV_64FC1, 1.0/255);

  
  Laplacian(greyImage, retval, CV_64FC1, ksize_);
  retval = abs(retval);

  return retval;
}

// --------- Start AverageCenterSurround ---------
CenterSurroundHistogram::~CenterSurroundHistogram() {}

CenterSurroundHistogram::CenterSurroundHistogram(
  const std::vector<double>& surroundScales,
  const std::string& distType)
  : surroundScales_(surroundScales) {
  if (distType == "chisq") {
    distType_ = CV_COMP_CHISQR;
  } else if (distType == "correl") {
    distType_ = CV_COMP_CORREL;
  } else if (distType == "intersect") {
    distType_ = CV_COMP_INTERSECT;
  } else if (distType == "bhattacharyya") {
    distType_ = CV_COMP_BHATTACHARYYA;
  } else {
    ROS_ERROR_STREAM("Invalid distance type: " << distType
                     << " defaulting to Chi Squared");
    distType_ = CV_COMP_CHISQR;
  }
}

Mat_<double> CenterSurroundHistogram::CalculateVisualUtility(const Mat& image,
                                                             double time) {
  ROS_FATAL("The visual utility for every pixel is not defined for this "
            "estimator.");
  return Mat_<double>();
}

bool CenterSurroundHistogram::InitBoxCalculator(const cv::Mat& image,
                                                double time) {
  // Get the image intensity since that's the feature we're going to use
  Mat tmpGreyImage;
  if (image.channels() != 1) {
    cvtColor(image, tmpGreyImage, CV_BGR2GRAY);
  } else {
    tmpGreyImage = image;
  }
  Mat_<uint8_t> greyImage(tmpGreyImage);

  // Calculate the integral histogram for the image
  integralHist_.reset(cv_utils::IntegralHistogram<int>::Calculate<uint8_t>(
    greyImage, 64, &std::pair<uint8_t, uint8_t>(0, 255)));

  return true;
}

double CenterSurroundHistogram::CalculateVisualUtilityOfBox(int x, int y,
                                                            int h, int w) {
  double bestScore = 0;
  for (vector<double>::const_iterator scaleI = surroundScales_.begin();
       scaleI != surroundScales_.end(); ++scaleI) {
    const double centerW = w * (*scaleI);
    const double centerH = h * (*scaleI);
    const int centerX1 = cvRound(x + (w - centerW) / 2.0);
    const int centerY1 = cvRound(y + (h - centerH) / 2.0);
    Mat_<float> centerHist = 
      integralHist_->GetHistInRegion(Rect(centerX1, centerY1,
                                          cvRound(centerW), cvRound(centerH)));
      
    const double surW = centerW * M_SQRT2;
    const double surH = centerH * M_SQRT2;
    const int surX1 = cvRound(x + (w - surW) / 2.0);
    const int surY1 = cvRound(y + (h - surH) / 2.0);
    Mat_<float> surHist =
      integralHist_->GetHistInRegion(Rect(surX1, surY1,
                                          cvRound(surW), cvRound(surH)));
    surHist -= centerHist;

    // Normalize the histograms
    const int centerArea = cvRound(centerH) * cvRound(centerW);
    centerHist /= centerArea;
    surHist /= (cvRound(surW) * cvRound(surH) - centerArea);
    
    double score = compareHist(centerHist, surHist, distType_);
    if (score > bestScore) {
      bestScore = score;
    }
  }
  return bestScore;
}

// --------- Start Objectness ------------
Objectness::~Objectness() {}

Objectness::Objectness() : impl_(true) {
  if (!impl_.Init()) {
    ROS_FATAL_STREAM("Couldn't not initialize the objectness wrapper");
  }
}

void Objectness::CalculateVisualUtility(const cv::Mat& image,
                                        const std::vector<cv::Rect>& rois,
                                        double time,
                                        std::vector<ROIScore>* vuOut) {
  lastRuntime_.reset(new double(0));
  impl_.CalculateObjectness(image, rois, vuOut, lastRuntime_.get());
}

cv::Mat Objectness::CalculateVisualUtility(const cv::Mat& image,
                                           int minX, int minY,
                                           int minW, int minH,
                                           int strideX, int strideY,
                                           double strideW, double strideH,
                                           bool fixAspect,
                                           double time,
                                           const cv::Mat& mask) {
  Mat scores;

  vector<double> widths;
  vector<double> heights;
  GetGridHeightsAndWidths(image, minX, minY, minW, minH, strideW, strideH,
                          fixAspect, &widths, &heights);

  // Build the output score matrix
  InitializeScoreGrid(image, minX, minY, strideX, strideY, fixAspect,
                      widths, heights, &scores);
  ROS_ASSERT(mask.empty() ||
             (mask.dims == scores.dims &&
              mask.size == scores.size));

  // Build up the list of boxes to evaluate
  vector<Rect> rois;
  for (unsigned int widthI = 0u; widthI < widths.size(); ++widthI) {
    for (int curX = minX; curX < image.cols-widths[widthI];) {
      for (int curY = minY; curY < image.rows;) {
        if (fixAspect) {
          if (curY + heights[widthI] <= image.rows) {
            rois.push_back(Rect(curX, curY, widths[widthI], heights[widthI]));
          }
        } else {
          for (unsigned int heightI = 0u;
               heightI < heights.size() && 
                 curY + heights[heightI] <= image.rows;
               ++heightI) {
            rois.push_back(Rect(curX, curY, widths[widthI],
                                heights[heightI]));
          }
        }
        curY += strideY;
      }
      curX += strideX;
    }
  }

  // Do the evaluation
  vector<ROIScore> scoreList;
  CalculateVisualUtility(image, rois, time, &scoreList);

  // Now load the results into the output grid
  int gridIdx[4]; // (s, x, y) or (w, h, x, y)
  for (vector<ROIScore>::const_iterator scoreI = scoreList.begin();
       scoreI != scoreList.end(); ++scoreI) {
    gridIdx[0] = (log(scoreI->second.width) - log(minW)) / log(strideW);
    gridIdx[scores.dims-2] = (scoreI->second.x - minX) / strideX;
    gridIdx[scores.dims-1] = (scoreI->second.y - minY) / strideY;
    if (!fixAspect) {
      gridIdx[1] = (log(scoreI->second.height) - log(minH)) / log(strideH);
    }
    ROS_ASSERT(gridIdx[0] >= 0 && gridIdx[1] >= 0 && gridIdx[2] >= 0 &&
               gridIdx[0] < scores.size[0] &&
               gridIdx[1] < scores.size[1] &&
               gridIdx[2] < scores.size[2]);
    scores.at<double>(gridIdx) = scoreI->first;
  }

  return scores;
}


Mat_<double> Objectness::CalculateVisualUtility(const Mat& image,
                                                double time) {
  ROS_FATAL("Visual utility for a pixel is not defined");
  return Mat_<double>::zeros(image.rows, image.cols);
}

// ---------- Start ROSBag ---------------
ROSBag::ResultKey::ResultKey(const std::string& _filename,
                             const cv::Rect _rect)
  : filenameHash(hash<string>()(_filename)), rect(_rect) {}
size_t ROSBag::ResultKeyHash::operator()(const ResultKey& s) const {
  size_t result = s.filenameHash;
  hash_combine(result, hash_value(s.rect));
  return result;
}

ROSBag::ROSBag(const string& filename) {
  rosbag::Bag bag;
  bag.open(filename, rosbag::bagmode::Read);

  CreateBaseEstimator(bag);

  LoadBagResults(bag);

  bag.close();
}

ROSBag::~ROSBag() {}

void ROSBag::CreateBaseEstimator(const rosbag::Bag& bag) {

  // Load all the parameters into ros params
  ros::NodeHandle handle("~ROSBag");
  rosbag::View view(bag, rosbag::TopicQuery("parameters"));
  BOOST_FOREACH(rosbag::MessageInstance const msg, view) {
    Parameter::ConstPtr param = msg.instantiate<Parameter>();
    if (param != NULL) {
      handle.setParam(param->name.data, param->value.data);
    }
  }

  // Create the visual utility estimator
  transformEstimator_.reset(CreateTransformEstimator(handle));
  baseEstimator_.reset(CreateVisualUtilityEstimator(handle,
                                                    *transformEstimator_));
  
  // Delete all the parameters
  rosbag::View view2(bag, rosbag::TopicQuery("parameters"));
  BOOST_FOREACH(rosbag::MessageInstance const msg, view2) {
    Parameter::ConstPtr param = msg.instantiate<Parameter>();
    if (param != NULL) {
      handle.deleteParam(param->name.data);
    }
  }
}

void ROSBag::LoadBagResults(const rosbag::Bag& bag) {
  rosbag::View view(bag, rosbag::TopicQuery("results"));
  BOOST_FOREACH(rosbag::MessageInstance const msg, view) {
    VisualUtilityEstimation::ConstPtr result =
      msg.instantiate<VisualUtilityEstimation>();
    if (result != NULL) {
      if (result->regions.size() != result->scores.size()) {
        ROS_ERROR_STREAM("The number of regions and scores don't match. "
                         "Aborting");
        continue;
      }
      for (unsigned int i = 0u; i < result->regions.size(); ++i) {
        const sensor_msgs::RegionOfInterest& region = result->regions[i];
        lut_[ResultKey(result->image, Rect(region.x_offset, region.y_offset,
                                           region.width, region.height))] = 
          result->scores[i];
      }
    }
  }
}

void ROSBag::CalculateVisualUtility(const cv::Mat& image,
                                    const std::vector<cv::Rect>& rois,
                                    double time,
                                    std::vector<ROIScore>* vuOut) {
  if (baseEstimator_.get()) {
    baseEstimator_->CalculateVisualUtility(image, rois, time, vuOut);
  }
}

cv::Mat_<double> ROSBag::CalculateVisualUtility(const cv::Mat& image,
                                                double time) {
  if (baseEstimator_.get()) {
    return baseEstimator_->CalculateVisualUtility(image, time);
  }
  return Mat_<double>::zeros(image.rows, image.cols);
}

void ROSBag::CalculateVisualUtility(const std::string& filename,
                                    const std::vector<cv::Rect>& rois,
                                    double time,
                                    std::vector<ROIScore>* vuOut) {
  ROS_ASSERT(vuOut);

  // Go through all the rois and look them up in the lookup table.
  for (vector<Rect>::const_iterator roi = rois.begin(); roi != rois.end();
       ++roi) {
    hash_map<ResultKey, float, ResultKeyHash>::const_iterator entry = 
      lut_.find(ResultKey(filename, *roi));
    if (entry != lut_.end()) {
      vuOut->push_back(ROIScore(entry->second, *roi));
    } else {
      vuOut->push_back(ROIScore(0.0, *roi));
    }
  }

  if (vuOut->size() != rois.size()) {
    ROS_ERROR_STREAM("We could not find entries in the LUT for all of the "
                     "rois. "
                     << rois.size() - vuOut->size()
                     << " entries were missing for file: "
                     << filename);
  }
}

// ---------- Start HOGDetector ---------------
HOGDetector::~HOGDetector() {}

HOGDetector::HOGDetector(const std::string& modelFile,
                         bool useDefaultPeopleDetector,
                         Size winStride,
                         bool doCache) {
  if (!impl_.InitModel(modelFile, useDefaultPeopleDetector,
                       -numeric_limits<double>::infinity(), // threshold
                       false, // doNMS
                       winStride,
                       doCache)) {
    ROS_FATAL_STREAM("Could not initialize HOG detector");
  }
}

void HOGDetector::CalculateVisualUtility(const cv::Mat& image,
                                         const std::vector<cv::Rect>& rois,
                                         double time,
                                         std::vector<ROIScore>* vuOut) {
  ROS_ASSERT(vuOut);

  vector<double> scores;
  vector<Rect> foundLocations;
  lastRuntime_.reset(new double(0));
  impl_.DetectObjects(image, rois, &foundLocations, &scores,
                       lastRuntime_.get());

  ROS_ASSERT(scores.size() == foundLocations.size());

  for (unsigned int i = 0u; i < foundLocations.size(); ++i) {
    vuOut->push_back(ROIScore(scores[i], foundLocations[i]));
  }
}

Mat_<double> HOGDetector::CalculateVisualUtility(const Mat& image,
                                                 double time) {
  ROS_FATAL("The visual utility for every pixel is not defined for this "
            "estimator.");
  return Mat_<double>();
}


// ---------- Start CascadeDetector ---------------
CascadeDetector::~CascadeDetector() {}
CascadeDetector::CascadeDetector(const std::string& modelFile){
  if (!impl_.Init(modelFile)) {
    ROS_FATAL_STREAM("Could not load the classifier defined in " << modelFile);
  }
}

Mat_<double> CascadeDetector::CalculateVisualUtility(const Mat& image,
                                                 double time) {
  ROS_FATAL("The visual utility for every pixel is not defined for this "
            "estimator.");
  return Mat_<double>();
}

void CascadeDetector::CalculateVisualUtility(
  const cv::Mat& image,
  const std::vector<cv::Rect>& rois,
  double time,
  std::vector<ROIScore>* vuOut) {

  ROS_ASSERT(vuOut);

  if (image.empty()) {
    return;
  }

  vector<int> scores;
  vector<Rect> foundLocations;
  lastRuntime_.reset(new double(0));
  impl_.DetectObjects(image, rois, &foundLocations, &scores,
                      lastRuntime_.get());

  ROS_ASSERT(scores.size() == rois.size());
  for (unsigned int i = 0u; i < scores.size(); ++i) {
    vuOut->push_back(ROIScore(scores[i], rois[i]));
  }
}

// ----------- Start ScaledDetectorWrapper -----------
ScaledDetectorWrapper::~ScaledDetectorWrapper() {}

bool ScaledDetectorWrapper::InitBoxCalculator(const cv::Mat& image,
                                              double time) {
  ros::WallTime startTime = ros::WallTime::now();

  // Resize the image and send it to the base estimator
  Mat scaledImage;
  cv::resize(image, scaledImage, cv::Size(), scaleFactor_, scaleFactor_);

  resizeTime_ = ros::WallTime::now() - startTime;

  return baseEstimator_->InitBoxCalculator(scaledImage, time);
}

double ScaledDetectorWrapper::CalculateVisualUtilityOfBox(int x, int y,
                                                          int h, int w) {
  // Scale the coordinates of the box and send to the base estimaor
  return baseEstimator_->CalculateVisualUtilityOfBox(
    cvRound(x * scaleFactor_),
    cvRound(y * scaleFactor_),
    cvRound(h * scaleFactor_),
    cvRound(w * scaleFactor_));
}

void ScaledDetectorWrapper::CalculateVisualUtility(
  const cv::Mat& image,
  const std::vector<cv::Rect>& rois,
  double time,
  std::vector<ROIScore>* vuOut) {
  ROS_ASSERT(vuOut);

  // Scale the input rois
  vector<Rect> scaledRois;
  unordered_map<Rect, const Rect*> roiMap;
  for (vector<Rect>::const_iterator roiI = rois.begin();
       roiI != rois.end();
       ++roiI) {
    scaledRois.push_back(Rect(cvRound(roiI->x * scaleFactor_),
                              cvRound(roiI->y * scaleFactor_),
                              cvRound(roiI->width * scaleFactor_),
                              cvRound(roiI->height * scaleFactor_)));
    roiMap[scaledRois.back()] = &(*roiI);
  }

  ros::WallTime startTime = ros::WallTime::now();

  // Resize the image
  Mat scaledImage;
  cv::resize(image, scaledImage, cv::Size(), scaleFactor_, scaleFactor_);
  resizeTime_ = ros::WallTime::now() - startTime;
  
  // Calculate the visual utility
  vector<ROIScore> scaledScores;
  startTime = ros::WallTime::now();
  baseEstimator_->CalculateVisualUtility(scaledImage,
                                         scaledRois,
                                         time,
                                         &scaledScores);
  ros::WallDuration baseTime = ros::WallTime::now() - startTime;

  // Scale the output rois
  for (vector<ROIScore>::const_iterator scoreI = scaledScores.begin();
       scoreI != scaledScores.end();
       ++scoreI) {
    vuOut->push_back(ROIScore(scoreI->first,
                              *roiMap[scoreI->second]));
  }

  // Do the timing properly
  lastRuntime_.reset(new double(0));
  *lastRuntime_ += resizeTime_.toSec();
  if (baseEstimator_->GetLastRuntime()) {
    *lastRuntime_ += *baseEstimator_->GetLastRuntime();
  } else {
    *lastRuntime_ += baseTime.toSec();
  }
  
}

cv::Mat_<double> ScaledDetectorWrapper::CalculateVisualUtility(
  const cv::Mat& image,
  double time) {
  ros::WallTime startTime = ros::WallTime::now();

  // Rescale the image
  Mat scaledImage;
  cv::resize(image, scaledImage, cv::Size(), scaleFactor_, scaleFactor_);

  ros::WallDuration resizeTime = ros::WallTime::now() - startTime;
  startTime = ros::WallTime::now();

  // Calculate the visual utility on the scaled image
  Mat_<double> scaledVU = baseEstimator_->CalculateVisualUtility(
    scaledImage, time);

  Mat_<double> retval;
  cv::resize(scaledVU, retval, image.size());

  ros::WallDuration baseTime = ros::WallTime::now() - startTime;

  // Store the timing data properly
  lastRuntime_.reset(new double(0));
  *lastRuntime_ += resizeTime.toSec();
  if (baseEstimator_->GetLastRuntime()) {
    *lastRuntime_ += *baseEstimator_->GetLastRuntime();
  } else {
    *lastRuntime_ += baseTime.toSec();
  }
  return retval;
}


// ----------- Start IntegralHOGDetector -----------
IntegralHOGDetector::IntegralHOGDetector(const std::string& modelFile,
                                         const cv::Size& winStride)
  : impl_(modelFile, winStride), hist_(), histSum_() {}

IntegralHOGDetector::~IntegralHOGDetector() {}

bool IntegralHOGDetector::InitBoxCalculator(const cv::Mat& image,
                                            double time) {
  hist_.reset(impl_.ComputeGradientIntegralHistograms(image, &histSum_));

  return true;
}

double IntegralHOGDetector::CalculateVisualUtilityOfBox(int x, int y, int h,
                                                        int w) {
  return impl_.ComputeScore(*hist_, histSum_, Rect(x, y, w, h));
}


cv::Mat_<double> IntegralHOGDetector::CalculateVisualUtility(
  const cv::Mat& image,
  double time) {
  ROS_FATAL("Per pixel evaluation is not implemented for this estimator");
  ROS_ASSERT(false);
  return cv::Mat_<double>();
}

// ----------- Start IntegralHOGCascade -----------
IntegralHOGCascade::IntegralHOGCascade(const std::string& modelFile,
                                        const cv::Size& winStride)
  : impl_(modelFile, winStride), hist_(), histSum_() {}

IntegralHOGCascade::~IntegralHOGCascade() {}

bool IntegralHOGCascade::InitBoxCalculator(const cv::Mat& image,
                                           double time) {
  hist_.reset(impl_.ComputeGradientIntegralHistograms(image, &histSum_));

  return true;
}

double IntegralHOGCascade::CalculateVisualUtilityOfBox(int x, int y, int h,
                                                       int w) {
  return impl_.ComputeScore(*hist_, histSum_, Rect(x, y, w, h));
}


cv::Mat_<double> IntegralHOGCascade::CalculateVisualUtility(
  const cv::Mat& image,
  double time) {
  ROS_FATAL("Per pixel evaluation is not implemented for this estimator");
  ROS_ASSERT(false);
  return cv::Mat_<double>();
}

} // namespace
