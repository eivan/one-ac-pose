CMAKE_MINIMUM_REQUIRED(VERSION 3.11)

# ==============================================================================
# Libraries
# ==============================================================================
add_subdirectory(OneACPose)

# ==============================================================================
# Executables
# ==============================================================================
if (ONEACPOSE_BUILD_SAMPLES)
  add_subdirectory(Samples)
endif()

# ==============================================================================
# Documentation
# ==============================================================================

# build doc if option is enabled
if (ONEACPOSE_BUILD_DOC)
	# check if Doxygen is installed
	find_package(Doxygen)
	if (DOXYGEN_FOUND)
		# set input and output files
		set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
		set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

		# request to configure the file
		configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
		message("Doxygen build started")

		# note the option ALL which allows to build the docs together with the application
		add_custom_target( doc_doxygen ALL
			COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
			COMMENT "Generating API documentation with Doxygen"
			VERBATIM )
		
		# install the documentation, too!
		install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc_doxygen DESTINATION "${include_dest}")
	else (DOXYGEN_FOUND)
	  message("Doxygen is needed to be installed to generate the doxygen documentation")
	endif (DOXYGEN_FOUND)
endif (ONEACPOSE_BUILD_DOC)