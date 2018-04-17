function(add_python_test test)
    add_test(NAME python_${test} COMMAND ${PYTHON_EXECUTABLE} ${test}.py)
    set_tests_properties(python_${test} PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/test")
endfunction(add_python_test)

find_package(PythonInterp REQUIRED)

configure_file(${CMAKE_SOURCE_DIR}/python/irbasis.py ${CMAKE_BINARY_DIR}/test/python COPYONLY)
file(GLOB_RECURSE TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
foreach(testfile ${TEST_FILES})
    configure_file(${testfile} ${CMAKE_BINARY_DIR}/test/python COPYONLY)
endforeach()

set(python_test_src check_ulx_vly Tnl)
foreach(test ${python_test_src})
    add_python_test(${test})
endforeach(test)
