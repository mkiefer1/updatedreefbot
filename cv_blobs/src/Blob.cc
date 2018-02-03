// Copyright 2010 ReefBot
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
//
// Blob.cpp
//
// A blob from a blob detection algorithm

#include <algorithm>
#include <assert.h>
#include "boost/lexical_cast.hpp"
#include <climits>
#include "opencv2/core/core.hpp"
#include "cv_bridge/CvBridge.h"
#include "cv_blobs/Blob.h"

using namespace cv;
using namespace boost;
using namespace std;
using reefbot_msgs::ImageRegion;

namespace cv_blobs {

ostream& operator<<(ostream& stream, const Blob& blob) {
  return blob.AsciiSerialize(stream);
}

// Constructor for a blob with a given id
Blob::Blob(int id, ConnectionType connectionType)
  : points_(), id_(id), connectionType_(connectionType),
    minX_(INT_MAX), minY_(INT_MAX), maxX_(INT_MIN), maxY_(INT_MIN) {}

bool Blob::AddPoint(const Point2i& point) {
  std::pair<BlobContainer::iterator, bool> result = 
    points_.insert(point);
  if (result.second) {
    if (point.x < minX_) {
      minX_ = point.x;
    }
    if (point.x > maxX_) {
      maxX_ = point.x;
    }
    if (point.y < minY_) {
      minY_ = point.y;
    }
    if (point.y > maxY_) {
      maxY_ = point.y;
    }
  }
  return result.second;
}

Rect Blob::GetBoundingBox() const {
  Point minPoint(minX(), minY());
  Point maxPoint(maxX()+1, maxY()+1);
  return Rect(minPoint, maxPoint);
}

ImageRegion::Ptr Blob::ToImageRegion() const {
  // Define the bounding box for the mask
  Rect bbox = GetBoundingBox();

  // Build a binary image mask for the blob
  Mat mask(Mat::zeros(bbox.height, bbox.width, CV_8UC1));
  for (BlobContainer::const_iterator i = begin(); i != end(); ++i) {
    mask.at<uchar>(*i - bbox.tl()) = 1;
  }

  // Finally convert the mask to the format needed by image region
  sensor_msgs::CvBridge bridge;
  IplImage iplMask = mask;
  sensor_msgs::Image::Ptr maskPtr =
    bridge.cvToImgMsg(&iplMask, "mono8");

  // Finally, create the object
  ImageRegion::Ptr imageRegion(new ImageRegion());
  imageRegion->bounding_box.x_offset = bbox.x;
  imageRegion->bounding_box.y_offset = bbox.y;
  imageRegion->bounding_box.width = bbox.width;
  imageRegion->bounding_box.height = bbox.height;
  imageRegion->mask = *maskPtr;

  return imageRegion;
}

Blob* Blob::copy() const {
  Blob* retval = new Blob(id_, connectionType_);
  retval->points_ = points_;
  retval->minX_ = minX_;
  retval->minY_ = minY_;
  retval->maxX_ = maxX_;
  retval->maxY_ = maxY_;

  return retval;
}

ostream& Blob::AsciiSerialize(ostream& stream) const {
  for (BlobContainer::iterator point = begin(); 
       point != end(); ++point) {
      stream << point->x << ',' << point->y << ';' ;
  }
  if (area() > 0) {
    stream << endl;
  }
  stream.flush();
  return stream;
}

shared_ptr<Blob> Blob::CreateFromStream(istream& stream, int id) {
  shared_ptr<Blob> retVal(new Blob(id, FOUR_CONNECTED));

  retVal->AsciiDeserialize(stream);

  return retVal;
}

istream& Blob::AsciiDeserialize(istream& stream) {
  Point2i point;
  string buf;

  // Keep going until we see the end of the line
  while(stream.peek() != '\n' && !stream.eof()) {
    // Pull out the x
    getline(stream, buf, ',');
    try {
      point.x = lexical_cast<int>(buf);
    } catch(bad_lexical_cast e) {
      throw ios_base::failure(string("Read an invalid value for x: ")
                              + buf);
    }
    buf.clear();

    // Look for the EOF because that would mean an error has occurred
    if (stream.eof()) {
      throw ios_base::failure("Unexpected end of stream");
    }

    // Pull out the y
    getline(stream, buf, ';');
    try {
      point.y = lexical_cast<int>(buf);
    } catch(bad_lexical_cast e) {
      throw ios_base::failure(string("Read an invalid value for y: ")
                              + buf);
    }
    buf.clear();

    AddPoint(point);
  }
  
  // eat the newline character
  stream.get();
  return stream;
}

Blob& Blob::operator+=(const cv::Point2i& point) {
  // First change the container of points
  BlobContainer newContainer;
  for(BlobContainer::iterator i = points_.begin();
      i != points_.end(); ++i) {
    newContainer.insert(*i + point);
  }
  points_.swap(newContainer);

  minX_ += point.x;
  maxX_ += point.x;
  minY_ += point.y;
  maxY_ += point.y;

  return *this;
}

}


