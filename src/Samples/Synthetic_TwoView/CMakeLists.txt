CMAKE_MINIMUM_REQUIRED(VERSION 3.11)

add_executable(Synthetic_TwoView
  main.cpp
)
target_link_libraries(Synthetic_TwoView
  OneACPose
  Eigen3::Eigen
  common
)
add_dependencies(Synthetic_TwoView
  OneACPose
  common
)