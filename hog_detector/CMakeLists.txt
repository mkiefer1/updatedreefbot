cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
include(${PROJECT_SOURCE_DIR}/../ROSCommon.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE Profile)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})

rosbuild_add_library(interpolation
  src/interpolation.cc)
target_link_libraries(interpolation
  gsl
  blas
)

rosbuild_add_library(hog_detector
  src/hog_detector.cc
  src/hog_detector_internal.cc)
target_link_libraries(hog_detector
opencv_objdetect
opencv_core
opencv_imgproc
cv_bridge
cv_utils
)

rosbuild_add_library(hog_detector_nodelet src/hog_detector_nodelet.cc)
target_link_libraries(hog_detector_nodelet
  hog_detector
  opencv_core
  opencv_imgproc
)

rosbuild_add_executable(hog_detector_node src/hog_detector_node.cc)
target_link_libraries(hog_detector_node
  hog_detector_nodelet
)

rosbuild_add_gtest(hog_detector_test test/hog_detector_test.cc)
target_link_libraries(hog_detector_test
  hog_detector
  opencv_highgui
  cv_utils
)


rosbuild_add_library(integral_hog_detector
  src/integral_hog_detector.cc
)
target_link_libraries(integral_hog_detector
  opencv_objdetect
  opencv_core
  opencv_ml
  opencv_highgui
  opencv_imgproc
  cv_utils
)

rosbuild_add_library(integral_hog_cascade
  src/integral_hog_cascade.cc
)
target_link_libraries(integral_hog_cascade
  integral_hog_detector
  opencv_core
  cv_utils
)

rosbuild_add_executable(integral_hog_trainer
  src/integral_hog_trainer.cc
)
target_link_libraries(integral_hog_trainer
  integral_hog_detector
  cv_utils
  opencv_core
  opencv_highgui
  gflags
)

rosbuild_add_library(integral_hog_cascade_trainer
  src/integral_hog_cascade_trainer.cc
)
target_link_libraries(integral_hog_cascade_trainer
  integral_hog_cascade
  interpolation
  opencv_core
  gsl
  blas
)

rosbuild_add_gtest(integral_hog_detector_test
  test/integral_hog_detector_test.cc)
target_link_libraries(integral_hog_detector_test
  integral_hog_detector
)

rosbuild_add_executable(ProfileIntegralHogDetector
  src/ProfileIntegralHogDetector.cc)
target_link_libraries(ProfileIntegralHogDetector
  integral_hog_detector
  gflags
)

rosbuild_add_executable(TrainCascadeUsingImages
  src/TrainCascadeUsingImages.cc)
target_link_libraries(TrainCascadeUsingImages
  integral_hog_cascade_trainer
  integral_hog_cascade
  integral_hog_detector
  gflags
)

rosbuild_add_executable(ExtractHardNegatives
  src/ExtractHardNegatives.cc)
target_link_libraries(ExtractHardNegatives
  gflags
  integral_hog_detector
  opencv_core
  opencv_highgui
)

rosbuild_add_gtest(interpolation_test
  test/interpolation_test.cc)
target_link_libraries(interpolation_test
  interpolation
)
