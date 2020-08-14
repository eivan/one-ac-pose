CMAKE_MINIMUM_REQUIRED(VERSION 3.11)

include_directories (BEFORE "${CMAKE_SOURCE_DIR}/third_party/VlFeatExtraction/src")
include_directories (BEFORE "${CMAKE_SOURCE_DIR}/third_party/VlFeatExtraction/third_party/")

find_package( OpenCV REQUIRED )
add_executable(Real_TwoView
  main.cpp
  solver_relative_pose_one_affine_and_relative_depth_minimal_svd.hpp
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