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
  PRIVATE
    Eigen3::Eigen
)