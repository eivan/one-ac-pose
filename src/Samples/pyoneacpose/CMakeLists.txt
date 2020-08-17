cmake_minimum_required(VERSION 2.8.4)

#project( pyoneacpose )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
  pybind11fetch
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.5.0
)

#FetchContent_GetProperties(pybind11fetch)
if(NOT googletest_POPULATED)
  #FetchContent_Populate(pybind11fetch)
  #add_subdirectory(${pybind11fetch_SOURCE_DIR} ${pybind11fetch_BINARY_DIR})
endif()

# Generate python module
#add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/graph-cut-ransac/lib/pybind11)

pybind11_add_module(pyoneacpose
  bindings.cpp
)

target_link_libraries(pyoneacpose
  PRIVATE
    Eigen3::Eigen
    OneACPose
    common
)

add_dependencies(pyoneacpose 
  OneACPose
  common
)