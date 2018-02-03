// Copyright 2011 Carnegie Mellon University
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
//
// VisualUtilityMosaic.h
//
// The visual utility mosaic is a compilation of the known visual
// utility. The visual utility measurements are stiched together over
// time.

#ifndef __VISUAL_UTILITY_MOSAIC_H__
#define __VISUAL_UTILITY_MOSAIC_H__

#include <opencv2/core/core.hpp>
#include <boost/scoped_ptr.hpp>
#include "TransformEstimator.h"
#include "cv_blobs/BlobResult.h"

namespace visual_utility {

class VisualUtilityMosaic;

// Abstract functors for the types of functions to maximize
struct PixelValFunc {
  // This one evaluates some function for each pixel, only using that pixel
  virtual double operator()(const double& val)=0;
};

struct RegionValFunc {
  // This functor evaluates some function for a region. The point of
  // interest (if applicable) is the center of the region.
  virtual double operator()(const cv::Mat& region)=0;
};

struct IntegralImageFunc {
  // This function evaluates some function using an integral
  // image. The function must evaluate based on a given center
  // point. The integral image can be accessed using the
  // GetIntegralValue function in the passed in mosaic
  virtual double operator()(const cv::Point2f& point,
                            const VisualUtilityMosaic& mosaic)=0;
};

// Abstract mosaic class to allow different implementations
class VisualUtilityMosaic {
public:
  // Takes ownership of transformEstimator
  VisualUtilityMosaic(TransformEstimator* transformEstimator,
                      int morphCloseSize,
                      double gaussSigma)
    : transformEstimator_(transformEstimator),
      morphCloseSize_(morphCloseSize),
      gaussSigma_(gaussSigma) {}
  virtual ~VisualUtilityMosaic();

  // Adds a visual utility frame to the mosaic with an optional
  // transform between the last frame and this one
  //
  // frame - The frame to add to the mosaic. Note: there is no
  // guarantee that the underlying data is copied
  // transform - The transform from the last frame to this one.
  //             Can be NULL if it is not known. In that case we 
  //             will try to calculate it.
  void AddFrame(cv::Mat_<double>& frame, const cv::Mat* transform);

  // Extracts a region from the visual utility mosaic. The coordinates
  // are setup so that (0,0) is the center of the previous frame. Regions
  // outside the mosaic will be filled with NaNs.
  virtual const cv::Mat_<double> ExtractRegion(const cv::Rect& roi) const=0;

  // These functions maximize some function over the visual utility mosaic.
  //
  // Inputs:
  // func - Function to maximize
  // maxLoc - Optional location of the maximum to return. (0,0) is the center
  // of the last frame added.
  //
  // Return: The maximum value found. Could be -numeric_limits<double>::infinity()
  virtual double MaximizeFunction(PixelValFunc* func,
                                  cv::Point2f* maxLoc) const=0;
  virtual double MaximizeFunction(RegionValFunc* func,
                                  const cv::Size2i& windowSize,
                                  cv::Point2f* maxLoc) const=0;
  virtual double MaximizeFunction(IntegralImageFunc* func,
                                  cv::Point2f* maxLoc) const=0;

  // Returns the value of the integral version of the visual utility
  // mosaic. The integral version defines each point as the sum of all
  // those points whose x or y coordinates are less than or equal to
  // the point given.
  //
  // Inputs:
  // point - Point to evaluate the integral image. (0,0) is
  // the center of the last frame added.
  virtual double GetIntegralValue(const cv::Point2f& point) const=0;

  // Returns the value at a given point
  virtual double GetValue(const cv::Point2f& point) const=0;

  // Calculate the sum of the visual utility mosaic across all space
  virtual double GetSum() const=0;

  // Calculate the sum in a box where (0,0) is the center of the last
  // frame added.
  virtual double GetSumInBox(const cv::Rect_<float>& box) const;

  // Get the size (in pixels) of the known areas in the visual utility mosaic
  virtual double GetSize() const=0;

  // Identifies all the connected components in the visual utility
  // mosaic where at point (x,y), f(x,y) > threshold
  virtual void FindConnectedComponents(
    PixelValFunc* func,
    double thresh,
    cv_blobs::BlobResult<double>* blobs) const=0;
  virtual void FindConnectedComponents(
    double thresh,
    cv_blobs::BlobResult<double>* blobs) const=0;
  

protected:
  boost::scoped_ptr<TransformEstimator> transformEstimator_;
  cv::Mat_<double> lastFrame_;

  // Function that actually adds the frame to the mosaic. This should
  // be rewritten by the subclass. transform will be empty only if we
  // couldn't figure it out.
  virtual void AddFrameImpl(cv::Mat_<double>& frame,
                            const cv::Mat& transform)=0;

  // Converts coordinates where (0,0) is the center of the last frame
  // added into coordinates where (0,0) is the top left of the last
  // frame.
  cv::Point2f ToFrameCoords(const cv::Point2f& point) const;
  // Converts in the opposite direction
  cv::Point2f ToBoresightCoords(const cv::Point2f& point) const;

private:
  int morphCloseSize_;
  double gaussSigma_;

  // Evil constructor
  VisualUtilityMosaic();

};

// A visual utility mosaic that just returns the last frame. It
// doesn't keep any state. So it's really not a mosaic.
class NullVUMosaic : public VisualUtilityMosaic {

public:
  NullVUMosaic(int morphCloseSize, double gaussSigma)
    : VisualUtilityMosaic(NULL, morphCloseSize, gaussSigma) {}
  virtual ~NullVUMosaic();

  // Extracts a region from the visual utility mosaic. The coordinates
  // are setup so that (0,0) is the center of the previous frame. Regions
  // outside the mosaic will be filled with NaNs.
  virtual const cv::Mat_<double> ExtractRegion(const cv::Rect& roi) const;

  // These functions maximize some function over the visual utility mosaic.
  //
  // Inputs:
  // func - Function to maximize
  // maxLoc - Optional location of the maximum to return. (0,0) is the center
  // of the last frame added.
  //
  // Return: The maximum value found
  virtual double MaximizeFunction(PixelValFunc* func,
                                 cv::Point2f* maxLoc) const;
  virtual double MaximizeFunction(RegionValFunc* func,
                                 const cv::Size2i& windowSize,
                                 cv::Point2f* maxLoc) const;
  virtual double MaximizeFunction(IntegralImageFunc* func,
                                  cv::Point2f* maxLoc) const;

  // Returns the value of the integral version of the visual utility
  // mosaic. The integral version defines each point as the sum of all
  // those points whose x or y coordinates are less than or equal to
  // the point given.
  //
  // Inputs:
  // point - Point to evaluate the integral image. (0,0) is
  // the center of the last frame added.
  virtual double GetIntegralValue(const cv::Point2f& point) const;

  // Returns the value at a given point
  virtual double GetValue(const cv::Point2f& point) const;

  // Calculate the sum of the visual utility mosaic across all space
  virtual double GetSum() const;

  // Get the size (in pixels) of the known areas in the visual utility mosaic
  virtual double GetSize() const;

  // Identifies all the connected components in the visual utility
  // mosaic where at point (x,y), f(x,y) > threshold
  virtual void FindConnectedComponents(
    PixelValFunc* func,
    double thresh,
    cv_blobs::BlobResult<double>* blobs) const;
  virtual void FindConnectedComponents(
    double thresh,
    cv_blobs::BlobResult<double>* blobs) const;

private:
  mutable boost::scoped_ptr<cv::Mat_<double> > integralImage_;
  mutable boost::scoped_ptr<double> mosaicSum_;

  virtual void AddFrameImpl(cv::Mat_<double>& frame,
                            const cv::Mat& transform);

};

}; // namespace

#endif // __VISUAL_UTILITY_MOSAIC_H__
