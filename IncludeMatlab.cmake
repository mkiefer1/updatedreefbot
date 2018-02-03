# Find the include path for matlab
execute_process(
  COMMAND matlab -nodisplay -r "display(fullfile(matlabroot, 'extern', 'include'));quit"
  RESULT_VARIABLE MATLAB_PROC
  OUTPUT_VARIABLE MATLAB_TEXT)
SET(FOUND_MATLAB_INCLUDES (${MATLAB_PROC} EQUAL 0))
if (NOT ${FOUND_MATLAB_INCLUDES})
  message(WARNING "Could not find the MATLAB include directory. Trying default")
  SET(MATLAB_INCLUDE_DIR "/usr/local/matlab/R2011/extern/include/")
else()
  STRING(REGEX MATCH "\n([/_0-9a-zA-Z]+/extern/include)" MATLAB_REG_RES ${MATLAB_TEXT})
  SET(MATLAB_INCLUDE_DIR ${CMAKE_MATCH_1})
endif()
message("Including " ${MATLAB_INCLUDE_DIR})
include_directories(${MATLAB_INCLUDE_DIR})


# Find the linker path for matlab
execute_process(
  COMMAND matlab -nodisplay -r "display(fullfile(matlabroot,'bin',computer('arch')));quit"
  RESULT_VARIABLE MATLAB_LIBS_PROC
  OUTPUT_VARIABLE MATLAB_LIBS_TEXT)
SET(FOUND_MATLAB_LIBS (${MATLAB_LIBS_PROC} EQUAL 0))
if (NOT ${FOUND_MATLAB_LIBS})
  message(WARNING "Could not find the MATLAB include directory. Trying the defult")
  SET(MATLAB_LIBS_DIR "/usr/local/matlab/R2011/bin/glnxa64")
else()
  STRING(REGEX MATCH "\n([/_0-9a-zA-Z]+/bin/[/_0-9a-zA-Z]+)" MATLAB_REG_RES ${MATLAB_LIBS_TEXT})
  SET(MATLAB_LIBS_DIR ${CMAKE_MATCH_1})
endif()
message("Linking to " ${MATLAB_LIBS_DIR})
link_directories(${MATLAB_LIBS_DIR})