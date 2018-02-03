// Python hook for finding objects in a blob 
//
// Copyright Reefbot 2011
//
// Author: Mark Desnoyer(mdesnoyer@gmail.com)
#include <Python.h>
#include <numpyconfig.h>
#include <numpy/arrayobject.h>
#include "cv_blobs/pyblob.h"

#include <boost/shared_ptr.hpp>
#include <ext/hash_set>
#include <opencv2/core/core.hpp>
#include <vector>

#include "cv_blobs/BlobResult-Inline.h"

#ifndef STD_NAMESPACE
  #ifdef __GNUC__
    #define STD_NAMESPACE __gnu_cxx
  #else // Windows
    #define STD_NAMESPACE std
  #endif
#endif

using namespace cv_blobs;
using namespace cv;

// Function that finds the blobs based on an object list.
//
// In python, this should be called like:
//
// blobs = pyblob.findblobs(points, nRows, nCols, connectionType)
//
// Where:
// points - tupe of x and y arrays ([x],[y]) (i.e. from the non-zero function)
// nRows, nCols - integers of the size of the image
// connectionType - integer for the type of connectivity. 0 for 4-connected 1 for 8 
// blobs - a list of list of x and y coordinates [[(x,y)]]
static PyObject* pyblob_findblobs(PyObject *self, PyObject*args) {

  PyObject* pointTuple;
  int nRows;
  int nCols;
  int connectionType;

  if (!PyArg_ParseTuple(args, "O!iii", &PyTuple_Type, &pointTuple, &nRows,
                        &nCols, &connectionType)) {
    return NULL;
  }

  if (PyTuple_Size(pointTuple) != 2) {
    PyErr_SetString(PyExc_TypeError, "Must be a tuple of arrays of size 2");
    return NULL;
  }

  PyObject* xObj = PyTuple_GetItem(pointTuple, 1);
  PyObject* yObj = PyTuple_GetItem(pointTuple, 0);

  PyObject* xArray = PyArray_FROM_OTF(xObj, NPY_INT32,
                                      NPY_IN_ARRAY | NPY_FORCECAST);
  if (xArray == NULL) {
    return NULL;
  }
  PyObject* yArray = PyArray_FROM_OTF(yObj, NPY_INT32,
                                      NPY_IN_ARRAY | NPY_FORCECAST);
  if (yArray == NULL) {
    Py_DECREF(xArray);
    return NULL;
  }

  bool inputFail = false;
  STD_NAMESPACE::hash_set<Point2i> cppPoints;
  if (PyArray_SIZE(xArray) == PyArray_SIZE(yArray)) {

    npy_int32* xData = static_cast<npy_int32*>(PyArray_DATA(xArray));
    npy_int32* yData = static_cast<npy_int32*>(PyArray_DATA(yArray));

    // Copy the points from the python list into a hash set needed by
    // the C++ library.
    int nPoints = PyArray_SIZE(xArray);
    for (int i = 0; i < nPoints; ++i) {
      npy_int32 x = *(xData + i);
      npy_int32 y = *(yData + i);
      
      cppPoints.insert(Point2i(x,y));
    }
  } else {
    PyErr_SetString(PyExc_TypeError, "Arrays in the tuple must be the same size");
    inputFail = true;
  }

  Py_DECREF(xArray);
  Py_DECREF(yArray);

  if (inputFail) {
    return NULL;
  }

  // Now find the blobs
  std::vector<boost::shared_ptr<Blob> > cppBlobs;
  FindBlobsFromPoints(&cppPoints, &cppBlobs,
                      static_cast<Blob::ConnectionType>(connectionType),
                      nRows, nCols);

  // Finally, build the output list of blobs
  PyObject* blobs = PyList_New(cppBlobs.size());
  int curBlobIdx = 0;
  for (std::vector<boost::shared_ptr<Blob> >::const_iterator blobI = 
         cppBlobs.begin();
       blobI != cppBlobs.end();
       ++blobI, curBlobIdx++) {
    // Build up a list of tuples for this blob
    PyObject* blob = PyList_New((*blobI)->area());

    int curPtIdx = 0;
    for (Blob::BlobContainer::const_iterator pointI = (*blobI)->begin();
         pointI != (*blobI)->end();
         ++pointI, ++curPtIdx) {
      PyObject* point = Py_BuildValue("(ii)", pointI->x, pointI->y);

      
      PyList_SET_ITEM(blob, curPtIdx, point);
    }

    PyList_SET_ITEM(blobs, curBlobIdx, blob);
  }

  return blobs;
}

// List of functions exported by this module
static PyMethodDef PyblobMethods[] = {
  {"findblobs", pyblob_findblobs, METH_VARARGS,
   "Finds the blobs based on an object list"},
  {NULL, NULL, 0, NULL} // Sentinel
};

// Initialize the module
PyMODINIT_FUNC initpyblob(void) {
  Py_InitModule("pyblob", PyblobMethods);
  import_array();
}
