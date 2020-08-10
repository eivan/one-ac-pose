# ==============================================================================
# Assemble the library target
# ==============================================================================

file(GLOB_RECURSE HDRS "*.hpp") 
file(GLOB_RECURSE SRCS "*.cpp") 

add_library(OneACPose
  ${HDRS}
  ${SRCS}
)

target_link_libraries(OneACPose
  PRIVATE
    Eigen3::Eigen
)