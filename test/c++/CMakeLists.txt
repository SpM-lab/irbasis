function(add_gtest test)
    set(source "${ARGV1}/${test}.cpp")
    set(gtest_src "${ARGV1}/gtest_main.cc;${ARGV1}/gtest-all.cc")

    add_executable(${test} ${source} ${gtest_src} ${header_files} dummy.cpp)
    target_link_libraries(${test} ${LINK_ALL})
    target_include_directories(${test} PRIVATE ${HDF5_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
    add_test(NAME ${test} COMMAND ${test} ${test_xml_output})
endfunction(add_gtest)

find_package(HDF5 REQUIRED)
find_package(Boost 1.54.0)

list(APPEND LINK_ALL ${CMAKE_THREAD_LIBS_INIT} ${HDF5_C_LIBRARIES})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} " CACHE STRING "Linker flags for executables" FORCE)

set(header_files ${CMAKE_SOURCE_DIR}/c++/irbasis.hpp)

configure_file(hdf5_test.h5 ${CMAKE_BINARY_DIR}/test/c++ COPYONLY)

#testing source files
set(unittest_src hdf5 multi_array interpolation)
foreach(test ${unittest_src})
    add_gtest(${test} ".")
endforeach(test)
