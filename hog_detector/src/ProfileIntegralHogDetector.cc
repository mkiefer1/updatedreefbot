// Program that estimates the timing for various components of an
// integral hog detector.
//
// Usage: ProfileIntegralHogDetector [options] <sampleImage0> <sampleImage1> ... <sampleImageN>
//
// The cache size is always half the block size and the block stride

#include "hog_detector/integral_hog_detector_inl.h"
#include <ros/ros.h>
#include <fstream>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>
#include <map>
#include <math.h>
#include <gflags/gflags.h>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <cstdlib>

// Options for what to measure
DEFINE_int32(min_block_size, 16, "Minimum block size to measure");
DEFINE_int32(max_block_size, 128, "Maximum block size to measure");
DEFINE_int32(block_size_stride, 8, "Stride in the block sizes to work with");
DEFINE_int32(min_subwin_size, 16, "Minimum subwindow size");
DEFINE_int32(max_subwin_size, 128, "Maximum subwindow size");
DEFINE_int32(subwin_stride, 16, "Stride for the subwindow size");

DEFINE_int32(win_stride, 8, "Stride that windows are moved around the image");
DEFINE_int32(block_stride, 8, "Stride for the blocks around the image");
DEFINE_int32(winW, 64, "Maximum width of a window");
DEFINE_int32(winH, 128, "Maximum height of a window");
DEFINE_int32(nbins, 9, "Number of orientation bins");
DEFINE_double(scale_stride, 1.10, "Stride for the scale of the windows");

// Options for when to finish
DEFINE_double(min_confidence, 0.05, "The minimum confidence, as measured by the standard error in percent of the time, of the timing estimate");

// Output file options
DEFINE_string(output_dir, ".", "Directory to place the outputs");
DEFINE_string(integral_hist_output, "integral_hist.txt",
              "Timing for calculating the integral histogram");
DEFINE_string(fill_cache_output, "fill_cache.txt",
              "Timing for filling the cache. Each line is <blockW>,<blockH>,<time>");
DEFINE_string(svm_eval_output, "svm_time.txt",
              "Timing for evaluating the svm value. Each line is <descriptorSize>,<time>");

using namespace std;
using namespace hog_detector;
using namespace boost;
using namespace cv;
using namespace boost::accumulators;
using namespace cv_utils;

typedef accumulator_set<double, stats<tag::mean> > TimeAccumulator;
typedef shared_ptr<TimeAccumulator> TimeAccumulatorPtr;
typedef map<Size, TimeAccumulatorPtr, bool(*)(const Size&, const Size&)>
  TimeAccumulatorMap;

// Predeclaration of functions
void FillCacheForAllSizes(const Mat& image,
                          const cv_utils::IntegralHistogram<float>* hist,
                          const cv::Mat_<float>* histSum,
                          HogBlockCache& cache,
                          double blockH,
                          double blockW);

// ------------- Done predeclaring --------------

void MeasureIntegralHistogramTime(const vector<string>& imageFiles) {
  ROS_INFO("Measuring the time to compute the integral histogram");

  IntegralHogDetector detector(Size(FLAGS_winW, FLAGS_winH),
                               Size(FLAGS_min_block_size,
                                    FLAGS_min_block_size),
                               Size(FLAGS_block_stride, FLAGS_block_stride),
                               Size(FLAGS_min_block_size / 2,
                                    FLAGS_min_block_size / 2),
                               FLAGS_nbins);

  accumulator_set<double, stats<tag::mean> >imageProcessingTime;

  for (vector<string>::const_iterator imageI = imageFiles.begin();
       imageI != imageFiles.end(); ++imageI) {
    ROS_INFO_STREAM("Processing " << *imageI);
    Mat image = imread(*imageI);

    accumulator_set<double, stats<tag::variance> > timeAccumulator;
    double curError = 1.0;
    while (curError > FLAGS_min_confidence) {
      scoped_ptr<IntegralHistogram<float> > integralHist;
      Mat_<float> histSum;

      ros::WallTime startTime = ros::WallTime::now();
      integralHist.reset(
        detector.ComputeGradientIntegralHistograms(image,&histSum));
      ros::WallDuration measTime = ros::WallTime::now() - startTime;

      timeAccumulator(measTime.toSec());

      if (boost::accumulators::count(timeAccumulator) > 3) {
        curError = sqrt(variance(timeAccumulator) /
                        boost::accumulators::count(timeAccumulator)) /
          boost::accumulators::mean(timeAccumulator);
      }
    }

    imageProcessingTime(boost::accumulators::mean(timeAccumulator));
  }

  ROS_INFO_STREAM("Average processing time to create the integral histogram: "
                  << boost::accumulators::mean(imageProcessingTime)
                  << " seconds");
  string integralHistFile = FLAGS_output_dir + "/" + 
    FLAGS_integral_hist_output;
  ROS_INFO_STREAM("Outputting processing time fo the integral histogram to "
                  << integralHistFile);
  fstream outStream(integralHistFile.c_str(),
                    ios_base::out | ios_base::trunc);
  outStream << boost::accumulators::mean(imageProcessingTime) << std::endl;
  outStream.close();
}

bool AreaOrder(const Size& left, const Size& right) {
  const int areaL = left.width * left.height;
  const int areaR = right.width * right.height;
  if (areaL == areaR) {
    return left.width < right.width;
  }
  return areaL < areaR;
}

void MeasureCacheFillTime(const vector<string>& imageFiles) {
  ROS_INFO("Measuring the time to fill the cache");

  // We're only valid if the strides are the same. Otherwise it is
  // more complicated to know when the cache is full.
  ROS_ASSERT(FLAGS_win_stride == FLAGS_block_stride);

  // A basic detector object
  IntegralHogDetector detector(Size(FLAGS_winW, FLAGS_winH),
                               Size(FLAGS_min_block_size,
                                    FLAGS_min_block_size),
                               Size(FLAGS_block_stride, FLAGS_block_stride),
                               Size(FLAGS_min_block_size / 2,
                                    FLAGS_min_block_size / 2),
                               FLAGS_nbins);

  // The accumulators for the time for each block size
  TimeAccumulatorMap imageProcessingTimes(AreaOrder);

  for (vector<string>::const_iterator imageI = imageFiles.begin();
       imageI != imageFiles.end(); ++imageI) {
    ROS_INFO_STREAM("Processing " << *imageI);
    Mat image = imread(*imageI);
    
    // Build the integral histogram
    Mat_<float> histSum;
    scoped_ptr<IntegralHistogram<float> > integralHist(
        detector.ComputeGradientIntegralHistograms(image,&histSum));

    for (int blockH = FLAGS_min_block_size;
         blockH <= FLAGS_max_block_size  && blockH <= FLAGS_winH;
         blockH += FLAGS_block_size_stride) {
      for (int blockW = FLAGS_min_block_size;
           blockW <= FLAGS_max_block_size && blockW <= FLAGS_winW;
           blockW += FLAGS_block_size_stride) {
        // Create the accumulator for this block size
        pair<TimeAccumulatorMap::iterator, bool> accIterTmp =
          imageProcessingTimes.insert(
            pair<Size, TimeAccumulatorPtr>(
              Size(blockW, blockH), TimeAccumulatorPtr(
                new TimeAccumulator())));
        TimeAccumulatorMap::iterator& accIter = accIterTmp.first;

        accumulator_set<double, stats<tag::variance> > curAccumulator;
        double curError = 1.0;
        while (curError > FLAGS_min_confidence) {
          // Create the cache
          HogBlockCache cache(FLAGS_nbins,
                              Size(FLAGS_win_stride, FLAGS_win_stride),
                              Size(FLAGS_block_stride, FLAGS_block_stride));

          ros::WallTime startTime = ros::WallTime::now();
          static const int N_TRIES = 20.0;
          for (int i = 0; i < N_TRIES; ++i) {
            FillCacheForAllSizes(image, integralHist.get(),
                                 &histSum, cache, blockH, blockW);
          }
          ros::WallDuration measTime = ros::WallTime::now() - startTime;

          curAccumulator(measTime.toSec() / N_TRIES);

          if (boost::accumulators::count(curAccumulator) > 3) {
            curError = sqrt(variance(curAccumulator) /
                            boost::accumulators::count(curAccumulator)) /
              boost::accumulators::mean(curAccumulator);
          }
        }
        
        (*accIter->second)(boost::accumulators::mean(curAccumulator));
      }
    }
  }

  string cacheFile = FLAGS_output_dir + "/" + FLAGS_fill_cache_output;
  ROS_INFO_STREAM("Outputting processing time for the cache fill to "
                  << cacheFile);
  fstream outStream(cacheFile.c_str(),
                    ios_base::out | ios_base::trunc);
  for (TimeAccumulatorMap::const_iterator i = imageProcessingTimes.begin();
       i != imageProcessingTimes.end();
       ++i) {
    outStream << i->first.width << ","
              << i->first.height << ","
              << boost::accumulators::mean(*i->second)
              << std::endl;
  }
  outStream.close();
    
}

void FillCache(const Mat& image,
               const cv_utils::IntegralHistogram<float>* hist,
               const cv::Mat_<float>* histSum,
               HogBlockCache& cache,
               double curFactor,
               double blockH,
               double blockW,
               double winHeight,
               double winWidth,
               double winStride) {
  cache.InitIfNecessary(hist, histSum, curFactor, curFactor,
                        Size(blockW, blockH),
                        Size(blockW/2, blockH/2));
  for (double y = 0; y < image.rows - winHeight; y += winStride) {
    for (double x = 0; x < image.cols - winWidth; x += winStride) {
      cache.GetBlock(Rect(round(x),
                          round(y),
                          round(winWidth),
                          round(winHeight)),
                     0, 0);
    }
  }
}

void FillCacheForAllSizes(const Mat& image,
                          const cv_utils::IntegralHistogram<float>* hist,
                          const cv::Mat_<float>* histSum,
                          HogBlockCache& cache,
                          double blockH,
                          double blockW) {
  
  double curWinHeight = FLAGS_winH;
  double curWinWidth = FLAGS_winW;
  double winStride = FLAGS_win_stride;
  double curFactor = 1.0;
  double scaleStride = FLAGS_scale_stride;
  while (curWinWidth < image.rows && curWinHeight < image.cols) {
    FillCache(image, hist, histSum, cache, curFactor, blockH, blockW,
              curWinHeight, curWinWidth, winStride);
    curWinWidth *= scaleStride;
    curWinHeight *= scaleStride;
    curFactor *= scaleStride;
    winStride *= scaleStride;
  }
}

// Returns the number of windows evaluated
int EvalAllWindows(const Mat& image,
                   const HogBlockCache& cache,
                   const Size& subWinSize,
                   const HogSVM& svm) {
  int nWindows = 0;

  int maxY = image.rows - FLAGS_winH;
  int maxX = image.cols - FLAGS_winW;
  Size blockSize(FLAGS_min_block_size, FLAGS_min_block_size);
  Size blockStride(FLAGS_block_stride, FLAGS_block_stride);
  for (float y = 0; y < maxY; y+= FLAGS_win_stride) {
    for (float x = 0; x < maxX; x += FLAGS_win_stride) {
      Rect subWin(round(x + 1.0*0),
                  round(y + 1.0*0),
                  round(subWinSize.width*1.0),
                  round(subWinSize.height*1.0));
                        
      HogBlockIterator blockIter =
        HogBlockIterator(subWin, &cache, subWinSize,
                         blockSize, blockStride);

      svm.predict<float>(&blockIter);

      ++nWindows;
    }
  }
  return nWindows;
}

int CalculateDescriptorSize(int nbins,
                            const Size& blockSize,
                            const Size& cellSize,
                            const Size& winSize,
                            const Size& blockStride) {
  return nbins * 
    (blockSize.width / cellSize.width) *
    (blockSize.height / cellSize.height) *
    ((winSize.width - blockSize.width) / blockStride.width + 1) *
    ((winSize.height - blockSize.height) / blockStride.height + 1);
}                   

void MeasurePredictTime(const vector<string>& imageFiles) {
  ROS_INFO("Measuring the time do an svm prediction");

  Size minBlockSize(FLAGS_min_block_size, FLAGS_min_block_size);
  Size minCellSize(FLAGS_min_block_size/2, FLAGS_min_block_size/2);
  Size winSize(FLAGS_winW, FLAGS_winH);
  Size blockStride(FLAGS_block_stride, FLAGS_block_stride);

  // A basic detector object
  IntegralHogDetector detector(winSize,
                               minBlockSize,
                               blockStride,
                               minCellSize,
                               FLAGS_nbins);

  // The SVM predictor. It's just random numbers
  vector<float> bigPredictor;
  int maxDescriptorSize = CalculateDescriptorSize(FLAGS_nbins,
                                                  minBlockSize,
                                                  minCellSize,
                                                  winSize,
                                                  blockStride);
  for (int i = 0; i < maxDescriptorSize; ++i) {
    bigPredictor.push_back(((double)rand()) / RAND_MAX);
  }
                                                  

  // The accumulators for the time for each block size
  TimeAccumulatorMap imageProcessingTimes(AreaOrder);

  for (vector<string>::const_iterator imageI = imageFiles.begin();
       imageI != imageFiles.end(); ++imageI) {
    ROS_INFO_STREAM("Processing " << *imageI);
    Mat image = imread(*imageI);

    // Build the integral histogram
    Mat_<float> histSum;
    scoped_ptr<IntegralHistogram<float> > integralHist(
        detector.ComputeGradientIntegralHistograms(image,&histSum));

    // Fill the cache
    HogBlockCache cache(FLAGS_nbins,
                        Size(FLAGS_win_stride, FLAGS_win_stride),
                        blockStride);
    FillCache(image, integralHist.get(), &histSum, cache, 1.0,
              FLAGS_min_block_size, FLAGS_min_block_size,
              FLAGS_winH, FLAGS_winW, FLAGS_win_stride);

    // Now loop through different subwindow sizes
    for (int subWinH = FLAGS_min_subwin_size;
         subWinH <= FLAGS_max_subwin_size && subWinH <= FLAGS_winH;
         subWinH += FLAGS_subwin_stride) {
      for (int subWinW = FLAGS_min_subwin_size;
         subWinW <= FLAGS_max_subwin_size && subWinW <= FLAGS_winW;
         subWinW += FLAGS_subwin_stride) {
        Size subWinSize(subWinW, subWinH);
        int descriptorSize = CalculateDescriptorSize(FLAGS_nbins,
                                                     minBlockSize,
                                                     minCellSize,
                                                     subWinSize,
                                                     blockStride);
        HogSVM svm;
        svm.SetDecisionVector(
          0.65, 
          vector<float>(bigPredictor.begin(),
                        bigPredictor.begin() + descriptorSize));

        accumulator_set<double, stats<tag::variance> > curAccumulator;
        double curError = 1.0;
        while (curError > FLAGS_min_confidence) {
          int nWindows = 0;
          ros::WallTime startTime = ros::WallTime::now();
          static const int N_TRIES = 5000 / descriptorSize;
          for (int i = 0; i < N_TRIES; ++i) {
            nWindows = EvalAllWindows(image, cache, subWinSize, svm);
          }
          ros::WallDuration measTime = ros::WallTime::now() - startTime;

          curAccumulator(measTime.toSec() / ((double)N_TRIES) / nWindows);

          if (boost::accumulators::count(curAccumulator) > 3) {
            curError = sqrt(variance(curAccumulator) /
                            boost::accumulators::count(curAccumulator)) /
              boost::accumulators::mean(curAccumulator);
          }
        }

        ROS_DEBUG_STREAM("Needed "
                         << boost::accumulators::count(curAccumulator)
                         << " runs to be confident of: "
                         << boost::accumulators::mean(curAccumulator)
                         << "s for window size " << descriptorSize);
        
        // Find the accumulator for this window size
        pair<TimeAccumulatorMap::iterator, bool> accIterTmp =
          imageProcessingTimes.insert(
            pair<Size, TimeAccumulatorPtr>(
              subWinSize, TimeAccumulatorPtr(
                new TimeAccumulator())));
        TimeAccumulatorMap::iterator& accIter = accIterTmp.first;


        (*accIter->second)(boost::accumulators::mean(curAccumulator));
      }
    }
  }

  string svmFile = FLAGS_output_dir + "/" + FLAGS_svm_eval_output;
  ROS_INFO_STREAM("Outputting processing time for the svm eval to "
                  << svmFile);
  fstream outStream(svmFile.c_str(),
                    ios_base::out | ios_base::trunc);
  for (TimeAccumulatorMap::const_iterator i = imageProcessingTimes.begin();
       i != imageProcessingTimes.end();
       ++i) {
    int descSize = CalculateDescriptorSize(FLAGS_nbins,
                                           minBlockSize,
                                           minCellSize,
                                           Size(i->first.width,
                                                i->first.height),
                                           blockStride);
    outStream << descSize << ","
              << i->first.width << ","
              << i->first.height << ","
              << boost::accumulators::mean(*i->second)
              << std::endl;
  }
  outStream.close();
}

int main(int argc, char **argv) {
  google::SetUsageMessage("[options] <sampleImage0> <sampleImage1> ... <sampleImageN>");
  google::ParseCommandLineFlags(&argc, &argv, true);

  ros::init(argc, argv, "ProfileIntegralHogDetector",
            ros::init_options::AnonymousName);
  ros::Time::init();

  // Grab the list of images
  vector<string> imageFiles;
  for (int i = 1; i < argc; ++i) {
    imageFiles.push_back(string(argv[i]));
  }

  MeasureIntegralHistogramTime(imageFiles);

  MeasureCacheFillTime(imageFiles);

  MeasurePredictTime(imageFiles);
}
