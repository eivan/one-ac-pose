# ==============================================================================
# Assemble the library target
# ==============================================================================

file(GLOB_RECURSE HDRS "*.hpp") 
file(GLOB_RECURSE SRCS "*.cpp") 

add_library(common
  ${HDRS}
  ${SRCS}
)

target_link_libraries(common
  Eigen3::Eigen
  GraphCutRANSAC
  ${OpenCV_LIBS}
  VlFeatExtraction
  cnpy
)

add_dependencies(common 
  OneACPose
  GraphCutRANSAC
  VlFeatExtraction
  cnpy
  common
)