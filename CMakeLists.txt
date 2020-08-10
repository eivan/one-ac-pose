CMAKE_MINIMUM_REQUIRED(VERSION 3.11)

project(OneACPose C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(ONEACPOSE_BUILD_SAMPLES "Build samples" ON)
option(ONEACPOSE_BUILD_DOC "Build documentation" OFF)
option(ONEACPOSE_USE_OPENMP "Use OpenMP for parallelization" OFF)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/third_party/)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# ==============================================================================
# Third party
# ==============================================================================
find_package( Eigen3 3.3.7 REQUIRED )
add_subdirectory(third_party)

# ==============================================================================
# Libraries & Executables
# ==============================================================================
add_subdirectory(src)