CMAKE_MINIMUM_REQUIRED(VERSION 3.11)

find_package( OpenCV REQUIRED )
add_executable(Real_TwoView
  main.cpp
)
target_link_libraries(Real_TwoView
  Eigen3::Eigen
  OneACPose
  common
)
add_dependencies(Real_TwoView 
  OneACPose
  common
)