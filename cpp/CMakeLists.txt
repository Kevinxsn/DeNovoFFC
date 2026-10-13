cmake_minimum_required(VERSION 3.16)
project(denovo_ffc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# All computation lives in one library, linked by the CLI and by the native
# tests, so a test exercises exactly the code the binary runs.
add_library(denovo_core STATIC
  src/pipeline.cpp
  src/lines.cpp
  src/masterset.cpp
  src/dag.cpp
  src/edges.cpp
  src/ffc.cpp
  src/json.cpp
  src/reconstruction.cpp
  src/optimal_paths.cpp
  src/rbdp.cpp
  src/rbdp_evidence.cpp
  src/combine.cpp
  src/gap_dictionary.cpp
)
target_include_directories(denovo_core PUBLIC include)

add_executable(denovo_ffc src/main.cpp)
target_link_libraries(denovo_ffc PRIVATE denovo_core)

add_executable(denovo_tests tests/test_native.cpp)
target_link_libraries(denovo_tests PRIVATE denovo_core)

# The pipeline expects the binaries at <repo>/bin/.
set_target_properties(denovo_ffc denovo_tests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../bin)
