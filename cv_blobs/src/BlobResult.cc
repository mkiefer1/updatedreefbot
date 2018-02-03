// Copyright 2011 ReefBot
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
//
// BlobResult.cpp
//
// Stores a set of results for a blob detection algorithm

#include "cv_blobs/BlobResult-Inline.h"
#include <assert.h>

namespace cv_blobs {

void FindBlobsFromPoints(STD_NAMESPACE::hash_set<Point2i>* points,
                         std::vector<boost::shared_ptr<Blob> >* output,
                         Blob::ConnectionType connectionType,
                         int nRows,
                         int nCols) {
  assert(points && output);

  // Build up the blobs one by one
  Rect frameBounds(0, 0, nCols, nRows);
  int curId = 0;
  stack<Point2i> curStack;
  Blob* curBlob = NULL;
  while (!points->empty()) {
    // See if we need to create a new blob
    if (curStack.empty()) {
      // Create a new blob
      curBlob = new Blob(curId++, connectionType);
      output->push_back(boost::shared_ptr<Blob>(curBlob));

      // Add the next point to the stack
      curStack.push(*points->begin());
    }

    // Take a point off the top of the stack and add it to the current blob
    const Point2i curElem = curStack.top();
    curStack.pop();
    STD_NAMESPACE::hash_set<Point2i>::iterator unusedIter =
      points->find(curElem);
    if (unusedIter != points->end() && curBlob->AddPoint(curElem)) {
      GetConnectedPoints(&curStack, frameBounds, curElem, connectionType);
    }
    points->erase(unusedIter);
  }
}

// Pushes any connected points that aren't in the blob onto the top
// of the stack.
//
// Inputs:
// stack - The stack to add points to
// frameBounds - Image bounds
// curPoint - The point around which to add adjacent ones to
void GetConnectedPoints(stack<Point2i>* stack,
                        const Rect& frameBounds,
                        const Point2i& curPoint,
                        Blob::ConnectionType connectionType) {
  assert(stack);

  if (connectionType == Blob::EIGHT_CONNECTED) {
    // Add the diagonals
    AddSafeConnected(stack, frameBounds, 
                     Point2i(curPoint.x - 1, curPoint.y - 1));
    AddSafeConnected(stack, frameBounds,
                     Point2i(curPoint.x - 1, curPoint.y + 1));
    AddSafeConnected(stack, frameBounds,
                     Point2i(curPoint.x + 1, curPoint.y - 1));
    AddSafeConnected(stack, frameBounds,
                     Point2i(curPoint.x + 1, curPoint.y + 1));
  }

  // Add the adjacent pixels
  AddSafeConnected(stack, frameBounds, Point2i(curPoint.x - 1, curPoint.y));
  AddSafeConnected(stack, frameBounds, Point2i(curPoint.x + 1, curPoint.y));
  AddSafeConnected(stack, frameBounds, Point2i(curPoint.x, curPoint.y - 1));
  AddSafeConnected(stack, frameBounds, Point2i(curPoint.x, curPoint.y + 1));
}

// Adds a point to the stack if it could be inside the frame.
void AddSafeConnected(stack<cv::Point2i>* stack,
                      const Rect& frameBounds,
                      const cv::Point2i& point) {
  if (frameBounds.contains(point)) {
    stack->push(point);
  }    
}

} // namespace
