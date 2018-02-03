cmake_minimum_required(VERSION 2.8)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
include(${PROJECT_SOURCE_DIR}/../ROSCommon.cmake)
include(${PROJECT_SOURCE_DIR}/../IncludeMatlab.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE Profile)

set(ROS_BUILD_STATIC_LIBS true)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# Specifies that we're using an environment that's cturtle
# compatible. In particular, the cv_bridge changed in Diamondback so
# if you want to use the new cv_bridge, comment out this define.
#add_definitions(-DCTURTLE)

#uncomment if you have defined messages
rosbuild_genmsg()
#uncomment if you have defined services
rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})

rosbuild_add_library(gnuplot src/gnuplot.cpp)

rosbuild_add_library(cvutils src/cvutils.cc)
target_link_libraries(cvutils
  opencv_core
  opencv_highgui
  gnuplot)

rosbuild_add_library(TransformEstimator src/TransformEstimator.cc)
target_link_libraries(TransformEstimator
  opencv_core
  opencv_imgproc
)

rosbuild_add_library(VisualUtilityFilter
  src/VisualUtilityEstimator.cc
  src/VisualUtilityMosaic.cc
  src/FrameEstimator.cc
  src/VisualUtilityFilter.cc
  src/VisualUtilityROSParams.cc
)
target_link_libraries(VisualUtilityFilter
  opencv_core
  opencv_imgproc
  TransformEstimator
  Objectness
  hog_detector
  integral_hog_detector
  integral_hog_cascade
  detailed_cascade_detector
  cv_utils
  gsl
  blas
  blobs
)

rosbuild_add_library(VisualUtilityEstimatorXMLParameters STATIC
  src/VisualUtilityEstimatorXMLParameters.cc
)
target_link_libraries(VisualUtilityEstimatorXMLParameters
  VisualUtilityFilter
)

rosbuild_add_library(TimingEstimator STATIC
  src/TimingEstimator.cc
)
target_link_libraries(TimingEstimator
  VisualUtilityFilter
  opencv_core
  opencv_highgui
)

rosbuild_add_executable(EstimateVUTiming
  src/EstimateVUTiming.cc
)
target_link_libraries(EstimateVUTiming
  TimingEstimator
  VisualUtilityFilter
  opencv_core
  opencv_highgui
)

rosbuild_add_executable(EvalVUOfImages
  src/EvalVUOfImages.cc
)
target_link_libraries(EvalVUOfImages
  TimingEstimator
  VisualUtilityFilter
  opencv_core
  opencv_highgui
  gflags
)
  

rosbuild_add_library(VisualUtilityEstimatorNodelet
  src/VisualUtilityEstimatorNodelet.cc
)
target_link_libraries(VisualUtilityEstimatorNodelet
  opencv_core
  VisualUtilityFilter
  cv_bridge
  cv_utils
)

rosbuild_add_executable(VisualUtilityEstimatorNode
  src/VisualUtilityEstimatorNode.cc
)
target_link_libraries(VisualUtilityEstimatorNode
  VisualUtilityEstimatorNodelet
)


rosbuild_add_library(VisualUtilityFilterNodelet
  src/VisualUtilityFilterNodelet.cc
)
target_link_libraries(VisualUtilityFilterNodelet
  opencv_core
  VisualUtilityFilter
  cv_bridge
  cv_utils
)

rosbuild_add_executable(VisualUtilityFilterNode
  src/VisualUtilityFilterNode.cc
)
target_link_libraries(VisualUtilityFilterNode
  VisualUtilityFilterNodelet
)

rosbuild_add_gtest(TransformEstimatorTest test/TransformEstimatorTest.cc)
target_link_libraries(TransformEstimatorTest
  TransformEstimator
  opencv_core
  opencv_imgproc
  opencv_highgui
  cv_utils)

rosbuild_add_gtest(GeometricTransformsTest test/GeometricTransformsTest.cc)
target_link_libraries(GeometricTransformsTest
  opencv_core
  opencv_imgproc)

rosbuild_add_gtest(SaliencyEstimatorTest test/SaliencyEstimatorTest.cc)
target_link_libraries(SaliencyEstimatorTest
  VisualUtilityFilter
  opencv_core
  opencv_highgui
  cvutils)

rosbuild_add_gtest(VisualUtilityEstimatorTest
  test/VisualUtilityEstimatorTest.cc)
target_link_libraries(VisualUtilityEstimatorTest
  VisualUtilityFilter
  opencv_core)

rosbuild_add_gtest(VisualUtilityMosaicTest test/VisualUtilityMosaicTest.cc)
target_link_libraries(VisualUtilityMosaicTest
  VisualUtilityFilter)

rosbuild_add_gtest(FrameEstimatorTest test/FrameEstimatorTest.cc)
target_link_libraries(FrameEstimatorTest
  VisualUtilityFilter)
