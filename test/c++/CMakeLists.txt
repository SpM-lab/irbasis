function(add_gtest test)
    set(source "${ARGV1}/${test}.cpp")
    set(gtest_src "${ARGV1}/gtest_main.cc;${ARGV1}/gtest-all.cc")

    add_executable(${test} ${source} ${gtest_src} ${header_files})
    target_link_libraries(${test} ${LINK_ALL})
    add_test(NAME ${test} COMMAND ${test} ${test_xml_output})
endfunction(add_gtest)

list(APPEND LINK_ALL ${CMAKE_THREAD_LIBS_INIT})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} " CACHE STRING "Linker flags for executables" FORCE)

set(header_files ${CMAKE_SOURCE_DIR}/c++/irbasis.hpp)

#testing source files
# Copy data-base files to build directory for testing
file(COPY ${CMAKE_SOURCE_DIR}/database DESTINATION ${CMAKE_BINARY_DIR}/c++ FILES_MATCHING  PATTERN "*.h5")
set(unittest_src interpolation)
foreach(test ${unittest_src})
    add_gtest(${test} ".")
endforeach(test)
