cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
include(${PROJECT_SOURCE_DIR}/../ROSCommon.cmake)

INCLUDE_DIRECTORIES(
  ${PROJECT_SOURCE_DIR}/../..
)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE RelWithDebInfo)

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

rosbuild_add_library(blobs
  src/Blob.cc
  src/BlobResult.cc)
target_link_libraries(blobs opencv_core)

add_executable(CreateBlobFrames src/CreateBlobFrames.cc)
target_link_libraries(CreateBlobFrames
  opencv_core
  opencv_highgui
  gflags
  blobs)

# All the tests go after this point
include_directories(src)

rosbuild_add_gtest(BlobTest test/BlobTest.cc)
TARGET_LINK_LIBRARIES(BlobTest blobs gtest opencv_core opencv_imgproc)

rosbuild_add_gtest(BlobResultTest test/BlobResultTest.cc)
TARGET_LINK_LIBRARIES(BlobResultTest blobs gtest opencv_core)


# All the python builds go after this point
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})
link_libraries(${PYTHON_LIBRARIES})
SET(PYTHON_EXECUTABLE /usr/bin/python)
find_package(Numpy REQUIRED)
include_directories(${PYTHON_NUMARRAY_INCLUDE_DIR})
rosbuild_add_library(pyblob src/cv_blobs/pyblob.cc)
set_target_properties(pyblob PROPERTIES OUTPUT_NAME pyblob PREFIX "" SUFFIX ".so")
target_link_libraries(pyblob
  blobs
  opencv_core
  )