CMAKE_MINIMUM_REQUIRED(VERSION 3.11)


# ==============================================================================
# Add subdirs
# ==============================================================================

set(CREATE_SAMPLE_PROJECT OFF) # turn gcransac sample project off
add_subdirectory(graph-cut-ransac)

add_subdirectory(VlFeatExtraction)

# ==============================================================================
# Add libraries explicitly
# ==============================================================================

find_package(ZLIB REQUIRED)

include_directories(${ZLIB_INCLUDE_DIRS})

add_library(cnpy
  cnpy/cnpy.cpp
  cnpy/cnpy.h
)

target_link_libraries(cnpy 
  ${ZLIB_LIBRARIES}
)

# ==============================================================================
# Organize project
# ==============================================================================

set_property(TARGET 
    # cnpy
    cnpy
    # GCRANSAC
    GraphCutRANSAC pygcransac #SampleProject
    # VlFeat
    VlFeatExtraction vlfeat
  PROPERTY FOLDER "third_party")