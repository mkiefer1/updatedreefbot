// Copyright 2010 ReefBot
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
//
// BlobResult.h
//
// Stores a set of results for a blob detection algorithm

#ifndef _CV_BLOBS_BLOB_RESULT__
#define _CV_BLOBS_BLOB_RESULT__

#include <boost/iterator/filter_iterator.hpp>
#include <boost/shared_ptr.hpp>
#include <stack>
#include <vector>
#include "opencv2/core/core.hpp"
#include "cv_blobs/Blob.h"

namespace cv_blobs {

template<typename ImageT>
class BlobResult {
public:
  // Constructor that finds blobs in an image.
  //
  // Inputs:
  // image - Image to find the blobs in
  // thresh - Threshold to determine if a pixel should be considered for blobs
  // connectionType - Determines what pixels are considered connected
  // aboveThresh - Are values above the threshold considered blobs?
  BlobResult(const cv::Mat_<ImageT>& image, ImageT thresh,
             Blob::ConnectionType connectionType=Blob::EIGHT_CONNECTED,
             bool aboveThresh=true);
  
  BlobResult() {};

  void FindBlobs(const cv::Mat_<ImageT>& image, ImageT thresh,
                 Blob::ConnectionType connectionType=Blob::EIGHT_CONNECTED,
                 bool aboveThresh=true);

  // Return the number of blobs found
  int nBlobs() const { return blobs_.size(); }

  // Get the nth blob from the set
  const Blob& GetBlob(int index) const { return *(blobs_[index]); }

  // Converts the blobs to an image which is 0 where there's no blob and index+1 for each blob
  cv::Mat ToImage() const;
  // Converts the blobs to an image that is 255 where there's a blob
  // and 0 otherwise
  cv::Mat ToBinaryImage() const { return ToImage() > 0; }

  // Filters the blobs for ones that meet the parameters of a
  // predicate.  The predicate must have an operator(const Blob&)
  // defined that returns true if the blob should be kept around.
  template<class Predicate>
  void Filter(Predicate pred) {
    std::vector<boost::shared_ptr<Blob> > newBlobs;
    for (unsigned int i = 0; i < blobs_.size(); ++i) {
      if (pred(*(blobs_[i]))) {
        newBlobs.push_back(blobs_[i]);
      }
    }
    blobs_.swap(newBlobs);
  }

  BlobResult<ImageT>* copy() const;

  BlobResult<ImageT>& operator+=(const cv::Point2i& point);
  BlobResult<ImageT>& operator-=(const cv::Point2i& point) {
    return operator+=(-point);
  }

  const cv::Size& ImageSize() { return size_; }

private:
  // Hide evil constructors
  BlobResult(const BlobResult<ImageT>& other);
  BlobResult<ImageT>& operator=(const BlobResult<ImageT>& other);

  std::vector<boost::shared_ptr<Blob> > blobs_;
  Blob::ConnectionType connectionType_;
  cv::Size size_; // Size of the image

  // Pushes any connected points that aren't in the blob onto the top
  // of the stack.
  //
  // Inputs:
  // stack - The stack to add points to
  // frameBounds - Image bounds
  // curPoint - The point around which to add adjacent ones to
  void GetConnectedPoints(std::stack<cv::Point2i>* stack,
                          const cv::Rect& frameBounds,
                          const cv::Point2i& curPoint) const;

  // Adds a point to the stack if it could be inside the frame.
  void AddSafeConnected(std::stack<cv::Point2i>* stack,
                        const cv::Rect& frameBounds,
                        const cv::Point2i& point) const;

  template <typename T> friend class BlobResultSerializer;
};

template <typename T>
class BlobResultSerializer {
public:

  // Serializes the blobs in the minimal way so that the first line of
  // the stream is the filename for the image we're talking about and
  // then each following line defines a blob.
  std::ostream& Serialize(std::ostream& stream,
                          const BlobResult<T>& blobs,
                          const std::string& imgName);

  // Deserializes a file that specifies the blob results for an
  // image. Optionally returns the filename of the image in imgName.
  boost::shared_ptr<BlobResult<T> > Deserialize(std::istream& stream,
                                                std::string* imgName,
                                                const std::string& dir="");

};
}

#endif // #ifndef _CV_BLOBS_BLOB_RESULT__
