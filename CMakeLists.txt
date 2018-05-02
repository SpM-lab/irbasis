cmake_minimum_required (VERSION 2.8.12)

# use only CXX compiler
project (irbasis CXX)

# find additional cmake module in cmake dir
#list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Print build type
if(NOT CMAKE_BUILD_TYPE)
    message("Using default build type: Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

### Dependencies
find_package(Threads)

### Eigen3
find_package(Eigen3 3.2.8 REQUIRED)

#include directories
#set(EIGEN3_INCLUDE_DIR "thirdparty/eigen3")
#set(IRLIB_LIBRARY_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR}/unsupported)
#include_directories(${IRLIB_LIBRARY_INCLUDE_DIRS})

# Build and enable tests
# testing setup
# enable_testing() must be called in the top-level CMakeLists.txt before any add_subdirectory() is called.
option(Testing "Enable testing" ON)

if (Testing)
  enable_testing(test)
  add_subdirectory(test)
endif()

#add_subdirectory(c++)
add_subdirectory(python)
