cmake_minimum_required(VERSION 2.8)
include(ExternalProject)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
include(${PROJECT_SOURCE_DIR}/../ROSCommon.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE Debug)

rosbuild_init()

set(GOIO_PREFIX ${EXTERNAL_BUILD_DIR}/goio)
set(GOIO_SRC_DIR ${GOIO_PREFIX}/src/goio)
set(GOIO_INSTALL_DIR ${GOIO_PREFIX}/install)
ExternalProject_Add(
  goio
  PREFIX ${GOIO_PREFIX}
  UPDATE_COMMAND "" # Stop it from updating because we want a static repo
  URL ${PROJECT_SOURCE_DIR}/GoIO-2.28.0.tar.gz
  SOURCE_DIR ${GOIO_SRC_DIR}
  CONFIGURE_COMMAND ${GOIO_SRC_DIR}/configure --prefix=${GOIO_INSTALL_DIR} --srcdir=${GOIO_SRC_DIR}
  BUILD_COMMAND make
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND make install
  INSTALL_DIR ${GOIO_INSTALL_DIR}
)

INCLUDE_DIRECTORIES(
  ${GOIO_INSTALL_DIR}/include/GoIO
)

LINK_DIRECTORIES(
  ${GOIO_INSTALL_DIR}/lib
)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
add_definitions(-DTARGET_OS_LINUX)
rosbuild_add_executable(device_check src/GoIO_DeviceCheck.cpp)
target_link_libraries(device_check GoIO)
add_dependencies(device_check goio)
