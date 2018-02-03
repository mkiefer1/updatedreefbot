cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

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

# Specifies that we're using an environment that's cturtle
# compatible. In particular, the cv_bridge changed in Diamondback so
# if you want to use the new cv_bridge, comment out this define.
add_definitions(-DCTURTLE)

rosbuild_add_library(detailed_cascade_detector
  src/detailed_cascade_detector.cc)
target_link_libraries(detailed_cascade_detector
  opencv_objdetect
  opencv_imgproc
)

rosbuild_add_library(cascade_detector_lib src/cascade_detector.cpp)
target_link_libraries(cascade_detector_lib 
  opencv_objdetect)

rosbuild_add_executable(cascade_detector src/cascade_detector_main.cpp)
target_link_libraries(cascade_detector cascade_detector_lib)

rosbuild_add_executable(estimate_detector_timing
  src/estimate_detector_timing.cc)
target_link_libraries(estimate_detector_timing
  detailed_cascade_detector
  opencv_objdetect
  opencv_imgproc
  opencv_highgui
)
