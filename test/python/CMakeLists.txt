function(add_python_test test)
    add_test(NAME python_${test} COMMAND ${PYTHON_EXECUTABLE} ${test}.py)
    set_tests_properties(python_${test} PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/test")
endfunction(add_python_test)

find_package(PythonInterp REQUIRED)

file(COPY ${CMAKE_SOURCE_DIR}/python/ DESTINATION ${CMAKE_BINARY_DIR}/test/python FILES_MATCHING PATTERN "*.py")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION ${CMAKE_BINARY_DIR}/test/python FILES_MATCHING PATTERN "*.py")

set(python_test_src interpolation)
foreach(test ${python_test_src})
    add_python_test(${test})
endforeach(test)
