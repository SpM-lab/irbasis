# Copy all *.h5 in database/
file(GLOB_RECURSE DATA_FILES "${CMAKE_SOURCE_DIR}/database/*.h5")
foreach(datafile ${DATA_FILES})
    configure_file(${datafile} ${CMAKE_BINARY_DIR}/test/ COPYONLY)
endforeach()

add_subdirectory(c++)
add_subdirectory(python)
