// Copyright 2010 ReefBot
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
//
// BlobResult.cpp
//
// Stores a set of results for a blob detection algorithm

#ifndef _CV_BLOBS_BLOB_RESULT_INLINE__
#define _CV_BLOBS_BLOB_RESULT_INLINE__

#include "cv_blobs/BlobResult.h"

#include <assert.h>
#include <boost/filesystem.hpp>
#include <ext/hash_set>
#include <iostream>
#include <stack>

#ifndef STD_NAMESPACE
  #ifdef __GNUC__
    #define STD_NAMESPACE __gnu_cxx
  #else // Windows
    #define STD_NAMESPACE std
  #endif
#endif

using namespace cv;
using namespace boost;
using namespace std;

namespace cv_blobs {

void FindBlobsFromPoints(STD_NAMESPACE::hash_set<Point2i>* points,
                         std::vector<boost::shared_ptr<Blob> >* output,
                         Blob::ConnectionType connectionType,
                         int nRows,
                         int nCols);

void GetConnectedPoints(stack<Point2i>* stack,
                        const Rect& frameBounds,
                        const Point2i& curPoint,
                        Blob::ConnectionType connectionType);

void AddSafeConnected(stack<cv::Point2i>* stack,
                      const Rect& frameBounds,
                      const cv::Point2i& point);

template <typename ImageT>
BlobResult<ImageT>::BlobResult(const cv::Mat_<ImageT>& image, ImageT thresh,
                               Blob::ConnectionType connectionType,
                               bool aboveThresh)
  : blobs_(), connectionType_(connectionType) {
  FindBlobs(image, thresh, connectionType, aboveThresh);
}

template <typename ImageT>
void BlobResult<ImageT>::FindBlobs(const cv::Mat_<ImageT>& image,
                                   ImageT thresh,
                                   Blob::ConnectionType connectionType,
                                   bool aboveThresh) {
  assert(image.channels() == 1);

  blobs_.clear();
  connectionType_ = connectionType;
  size_ = image.size();

  // Get the list of points that could be considered blobs
  STD_NAMESPACE::hash_set<Point2i> unusedPoints;
  for (int row = 0; row < image.rows; ++row) {
    const ImageT* rowPtr = image[row];
    for (int col = 0; col < image.cols; ++col) {
      if ( (aboveThresh && rowPtr[col] > thresh) || 
           (!aboveThresh && rowPtr[col] < thresh) ) {
        unusedPoints.insert(Point2i(col, row));
      }
    }
  }

  FindBlobsFromPoints(&unusedPoints, &blobs_, connectionType_,
                      image.rows, image.cols);      
}

template <typename ImageT>
Mat BlobResult<ImageT>::ToImage() const {
  Mat_<uchar> retval= Mat_<uchar>::zeros(size_);

  int curIdx = 1;
  for (vector<shared_ptr<Blob> >::const_iterator blobI = blobs_.begin();
       blobI != blobs_.end(); ++blobI, ++curIdx) {
    for (Blob::BlobContainer::const_iterator i = (*blobI)->begin(); 
         i != (*blobI)->end(); ++i) {
      retval(*i) = curIdx;
    }
  }

  return retval;
}

template <typename ImageT>
BlobResult<ImageT>* BlobResult<ImageT>::copy() const {
  BlobResult<ImageT>* retval = new BlobResult<ImageT>();
  retval->connectionType_ = connectionType_;
  retval->size_ = size_;

  for (vector<shared_ptr<Blob> >::const_iterator blobI = blobs_.begin();
       blobI != blobs_.end(); ++blobI) {
    retval->blobs_.push_back(shared_ptr<Blob>((*blobI)->copy()));
  }
  return retval;
}

template <typename ImageT>
BlobResult<ImageT>& BlobResult<ImageT>::operator+=(const Point2i& point) {
  for (vector<shared_ptr<Blob> >::iterator blobI = blobs_.begin();
       blobI != blobs_.end(); ++blobI) {
    **blobI += point;
  }
  return *this;
}

template <typename T>
ostream& BlobResultSerializer<T>::Serialize(ostream& stream,
                                            const BlobResult<T>& blobs,
                                            const string& imgName) {
  stream << imgName << endl;
  for (int i = 0; i < blobs.nBlobs(); i++) {
    blobs.GetBlob(i).AsciiSerialize(stream);
  }
  stream.flush();
  return stream;
}

template <typename T>
shared_ptr<BlobResult<T> > 
BlobResultSerializer<T>::Deserialize(istream& stream,
                                     string* imgName,
                                     const string& dir) {
  // Read the image name from the file
  string buf;
  getline(stream, buf);
  if (buf.size() > 0) {
    filesystem::path imgFile;
    if (buf[0] == '/') {
      imgFile = buf;
    } else {
      imgFile = filesystem::path(dir) / buf;
    }

    if (!filesystem::exists(imgFile)) {
      throw ios_base::failure(
        string("Image file specied in the blob file is not acessible: ") +
        imgFile.string());
    }

    if (imgName != NULL) {
      *imgName = imgFile.string();
    }
  }
  
  // Now build up the BlobResult object, one at a time
  shared_ptr<BlobResult<T> > ret(new BlobResult<T>());
  int id = 0;
  while(!stream.eof() && !stream.fail() && stream.peek() != EOF) {
    ret->blobs_.push_back(Blob::CreateFromStream(stream, id));
    id++;
  }
  if (stream.fail()) {
    throw ios_base::failure("Error reading blob description");
  }

  return ret;
}



}

#endif
