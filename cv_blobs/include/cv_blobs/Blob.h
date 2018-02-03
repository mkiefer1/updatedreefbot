// Copyright 2010 ReefBot
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
//
// Blob.h
//
// A blob from a blob detection algorithm

#ifndef _CV_BLOBS_BLOB__
#define _CV_BLOBS_BLOB__

#ifndef STD_NAMESPACE
  #ifdef __GNUC__
    #define STD_NAMESPACE __gnu_cxx
  #else // Windows
    #define STD_NAMESPACE std
  #endif
#endif

#include <boost/shared_ptr.hpp>
#include <ext/hash_set>
#include <iostream>
#include <stack>
#include "opencv2/core/core.hpp"

#include "reefbot_msgs/ImageRegion.h"

namespace STD_NAMESPACE {
template <>
struct hash<cv::Point2i> {
  size_t operator()(const cv::Point2i& point) const {
    int hashVal = 21;
    hashVal = point.x + hashVal*31;
    hashVal = point.y + hashVal*31;
    return hashVal;
  }
};
}

namespace cv_blobs {

class Blob {
public:
  typedef enum {
    FOUR_CONNECTED=0, // Only adjacent pixels are connected
    EIGHT_CONNECTED=1 // Adjacent and diagonal pixels are connected
  } ConnectionType;

  typedef STD_NAMESPACE::hash_set<cv::Point2i> BlobContainer;

  // Creates a blob with a given ID and connection type.
  Blob(int id, ConnectionType connectionType);

  // Adds a point to the list of points in this blob
  //
  // Inputs:
  // point - point to add
  //
  // Output: true if the point was added and wasn't there before
  bool AddPoint(const cv::Point2i& point);

  // Returns true if the blob contains a given point
  bool Contains(const cv::Point2i& contains) const {
    return points_.count(contains) > 0;
  }

  // Getters
  int minX() const { return minX_;}
  int maxX() const { return maxX_;}
  int minY() const { return minY_;}
  int maxY() const { return maxY_;}
  int id() const { return id_;}

  // Iterator for the points in the blob. Returns cv::Point2i objects
  BlobContainer::const_iterator begin() const { return points_.begin(); }
  BlobContainer::const_iterator end() const { return points_.end(); }

  // Determines the area of the blob in pixels
  int area() const { return points_.size(); }

  cv::Rect GetBoundingBox() const;

  // Converts a blob to an ImageRegion object. Caller takes ownership
  // of the new ImageRegion object.
  reefbot_msgs::ImageRegion::Ptr ToImageRegion() const;

  // For serializing the deserializing

  // Serializes the blob into a list of locations on a line. It is in
  // the format "x,y;x,y;x,y;\n"
  std::ostream& AsciiSerialize(std::ostream& stream) const;

  // Factory function to read in the blob from an ascii stream defined
  // by a line of the form: "x,y;x,y;x,y;\n"
  static boost::shared_ptr<Blob> CreateFromStream(std::istream& stream,
                                                  int id);

  Blob* copy() const;

  // Operator overloading
  bool operator==(const Blob& other) const {
    return points_ == other.points_;
  }
  bool operator!=(const Blob& other) const {
    return !(*this == other);
  }

  // Operator overloading to change the coordinates of the points in the blob
  Blob& operator+=(const cv::Point2i& point);
  Blob& operator-=(const cv::Point2i& point) {
    return operator+=(-point);
  }
  

private:
  // The image indices specifying this blob
  BlobContainer points_; 
  int id_;
  ConnectionType connectionType_;
  int minX_;
  int minY_;
  int maxX_;
  int maxY_;

  // Fills in the blob from an ascii stream defined by a line of the form:
  // "x,y;x,y;x,y;\n"
  std::istream& AsciiDeserialize(std::istream& stream);

  // Hide all evil constructors
  Blob();
  Blob(const Blob& other);
};

std::ostream& operator<<(std::ostream& stream, const Blob& blob);

}


#endif // #ifndef _CV_BLOBS_BLOB__
