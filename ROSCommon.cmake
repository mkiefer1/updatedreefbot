# We need pthread's
if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC")
endif()

# Add a define so that the code can know if we're building the pure debug mode
SET( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG" )
SET( CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DDEBUG" )

# Define new build type for profiling
SET( CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS_RELEASE} -g" CACHE STRING
  "Flags used by the C++ compiler during profile builds")
SET( CMAKE_C_FLAGS_PROFILE "${CMAKE_C_FLAGS_RELEASE} -g" CACHE STRING
    "Flags used by the C compiler during profile builds."
    FORCE )
SET( CMAKE_EXE_LINKER_FLAGS_PROFILE
    "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -lprofiler" CACHE STRING
    "Flags used for linking binaries during profile builds."
    FORCE )
SET( CMAKE_SHARED_LINKER_FLAGS_PROFILE
    "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -lprofiler" CACHE STRING
    "Flags used by the shared libraries linker during profile builds."
    FORCE )
MARK_AS_ADVANCED(
    CMAKE_CXX_FLAGS_PROFILE
    CMAKE_C_FLAGS_PROFILE
    CMAKE_EXE_LINKER_FLAGS_PROFILE
    CMAKE_SHARED_LINKER_FLAGS_PROFILE )

# Update the documentation string of CMAKE_BUILD_TYPE for GUIs
SET( CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Profile."
    FORCE )

set(EXTERNAL_BUILD_DIR ${PROJECT_BINARY_DIR}/external/ )