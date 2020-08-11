CMAKE_MINIMUM_REQUIRED(VERSION 3.11)

add_executable(Real_TwoView
  main.cpp
)
target_link_libraries(Real_TwoView
  OneACPose
  Eigen3::Eigen
  ${OpenCV_LIBS}
  VlFeatExtraction
  cnpy
  common
)
add_dependencies(Real_TwoView 
  VlFeatExtraction
  OneACPose
  cnpy
  common
)