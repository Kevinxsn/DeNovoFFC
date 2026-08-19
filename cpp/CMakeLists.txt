cmake_minimum_required(VERSION 3.16)
project(denovo_mms CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_executable(denovo_mms
  src/main.cpp
  src/pipeline.cpp
  src/lines.cpp
  src/masterset.cpp
  src/dag.cpp
  src/edges.cpp
  src/ffc.cpp
)
target_include_directories(denovo_mms PRIVATE include)

# The pipeline expects the binary at <repo>/bin/denovo_mms.
set_target_properties(denovo_mms PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../bin)
