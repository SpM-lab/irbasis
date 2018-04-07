file(COPY ${CMAKE_SOURCE_DIR}/database/ DESTINATION ${CMAKE_BINARY_DIR}/test FILES_MATCHING  PATTERN "*.h5")

add_subdirectory(c++)
add_subdirectory(python)
