CMAKE_MINIMUM_REQUIRED(VERSION 3.11)


include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories (BEFORE "${CMAKE_SOURCE_DIR}/third_party/VlFeatExtraction/src")
include_directories (BEFORE "${CMAKE_SOURCE_DIR}/third_party/VlFeatExtraction/third_party/")

add_subdirectory(common)
add_subdirectory(Synthetic_TwoView)
add_subdirectory(Real_TwoView)

set_property(TARGET 
    common
    Synthetic_TwoView
    Real_TwoView
  PROPERTY FOLDER "Samples")