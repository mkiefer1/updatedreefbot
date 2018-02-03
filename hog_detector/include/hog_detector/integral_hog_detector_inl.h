// A HOG detector built on integral histograms so that they are fast
// to compute
//
// Copyright 2012 Mark Desnoyer (mdesnoyer@gmail.com

#ifndef __HOG_DETECTOR_INTEGRAL_HOG_DETECTOR__INL_H__
#define __HOG_DETECTOR_INTEGRAL_HOG_DETECTOR__INL_H__

#include "hog_detector/integral_hog_detector.h"
#include <opencv2/core/core.hpp>
#include <ros/ros.h>

namespace hog_detector {

inline const float* HogBlockCache::GetBlock(const cv::Rect& win, int blockX,
                                            int blockY) const {
  int xIdx = round((win.x / xFactor_ + blockX) / cacheStride_.width);
  int yIdx = round((win.y / yFactor_ + blockY) / cacheStride_.height);

  ROS_ASSERT(xIdx < flags_.cols && yIdx < flags_.rows);

  // See if we've already calculated this block
  uchar* flagPtr = flags_.ptr<uchar>(yIdx, xIdx);
  float* blockPtr = blockCache_.ptr<float>(yIdx, xIdx, 0);
  if (*flagPtr) {
    return blockPtr;
  }
  *flagPtr = 1;
  return CalculateBlock(win, blockX, blockY, blockPtr);
}

inline const float* HogBlockIterator::operator*() const {
  return cache_->GetBlock(win_, curBlockX_, curBlockY_);
}

template<typename T>
T HogSVM::predict(HogBlockIterator* blockIter,
                  bool returnDFVal) const {
  T score = bias_;

  ROS_ASSERT(blockIter);

  const float* vectorPtr = &detectorVec_[0];
  int blockHistSize = blockIter->blockHistSize();
  int i;
  for (; !blockIter->isDone(); ++(*blockIter)) {
    const float* curBlock = **blockIter;
    for (i = 0; i <= blockHistSize - 4; i+=4) {
      score += curBlock[i]*vectorPtr[i] + 
        curBlock[i+1]*vectorPtr[i+1] + 
        curBlock[i+2]*vectorPtr[i+2] + 
        curBlock[i+3]*vectorPtr[i+3];
    }
    for (; i < blockHistSize; ++i) {
      score += curBlock[i]*vectorPtr[i];
    }
    vectorPtr += blockHistSize;
  }

  return returnDFVal ? score : static_cast<T>(score > 0);
}

}

#endif //__HOG_DETECTOR_INTEGRAL_HOG_DETECTOR__INL_H__
