// Copyright 2012 Carnegie Mellon University
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
//
// VisualUtilityROSParams.h
//
// Utilities to build the objects of the visual utility pipeline from
// ROS parameters.
//
// All of the routines return pointers to newly created objects that the
// caller must take ownership of.
//
// The parameters are relative to the input handle and options are:

// ------------ TransformEstimator options -------------
// affine_max_iterations - Max iterators for the AffineTransformEstimator
// min_affine_precision - Min precision required for AffineTransformEstimator
// affine_scaling_factor - Multiple to shrink the image to make affine
//                         transform estimation faster, but potentially
//                         less accurate. e.g. 2.0 will cut each image
//                         dimsion in half.

// ------------ VisualUtilityEstimator options -----------
// vu_estimator - String specifying the type of estimator to use
// vu_estimator_wrapper - String specifying the type of wrapper to use to 
//                        evaluate the estimator for a window. An empty 
//                        string means not used.

// ------------ VisualUtilityMosaic options ----------
// morph_close_size - When smoothing, the size of the morphological close
//                    operation.
// gauss_sigma - When smoothing, the sigma factor of the gaussian per second

// ------------ FrameEstimator options ----------
// frame_estimator - String specifying the class name of the frame estimator
//                   to use.
// xframesize - For fixed frame size techniques, the x dimension
// yframesize - For fixed frame size techniques, the y dimension
// frame_exampansion - Factor to expand the frame dimensions after the
//                     tight frame is found.
// min_frame_area - The minimum area captured in the frame to include that
//                  frame. Used to filter regions that are too small.
// min_entropy - With the relative entropy techniques, the minimum relative
//               entropy of a region needed in order to be returned.

#ifndef __VISUAL_UTILITY_ROS_PARAMS_H__
#define __VISUAL_UTILITY_ROS_PARAMS_H__

#include <ros/ros.h>
#include "VisualUtilityEstimator.h"
#include "VisualUtilityMosaic.h"
#include "FrameEstimator.h"
#include "TransformEstimator.h"

namespace visual_utility {

TransformEstimator* CreateTransformEstimator(ros::NodeHandle handle);

VisualUtilityEstimator* CreateVisualUtilityEstimator(
  ros::NodeHandle handle,
  const TransformEstimator& transformEstimator);

VisualUtilityMosaic* CreateVisualUtilityMosaic(
  ros::NodeHandle handle,
  const TransformEstimator& transformEstimator);

FrameEstimator* CreateFrameEstimator(ros::NodeHandle handle);

} // namespace

#endif // __VISUAL_UTILITY_ROS_PARAMS_H__
