// Copyright 2010 ReefBot
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
//
// BlobFilters.h
//
// A bunch of predefined prediates that can be used by BlobResult::Filter
//
// These are designed to take advantage of the STL binary predicates
// for comparisons like:
//
// equal_to
// not_equal_to
// less
// less_equal
// greater
// greater_equal
//
// So, for example,  to create an operator that returns true if the
// area is less than 500, use:
// AreaComparable<less<int> >(500)

#ifndef _CV_BLOBS_BLOB_FILTERS__
#define _CV_BLOBS_BLOB_FILTERS__

#include "cv_blobs/Blob.h"

namespace cv_blobs {

// A predicate that compares the area to a threshold. For example, to
// create an operator that returns true if the area is less than 500,
// use:
// AreaCompare<less<int> >(500)
template<typename ComparePred, typename T>
class AreaCompare {
public:
  AreaCompare(T thresh) : thresh_(thresh) {}
  bool operator()(const Blob& blob) {
    return ComparePred()(blob.area(), thresh_);
  }

private:
  T thresh_;

};

// A predicate that compares the size of the bounding box around the
// blob to a threshold.
template<typename ComparePred, typename T>
class BoxAreaCompare {
public:
  BoxAreaCompare(T thresh) : thresh_(thresh) {}
  bool operator()(const Blob& blob) {
    T area = static_cast<T>(blob.maxX() - blob.minX()) * 
      (blob.maxY() - blob.minY());
    return ComparePred()(area, thresh_);
  }

private:
  T thresh_;

};

}

#endif // #ifndef _CV_BLOBS_BLOB_FILTERS__
