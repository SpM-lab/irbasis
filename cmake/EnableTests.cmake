# 
# enables testing with google test
# provides add_gtest(test) and add_gtest_release(test) commands
#

# check xml output
option(TestXMLOutput "Output tests to xml" OFF)

# custom function to add gtest with xml output
# arg0 - test (assume the source is ${test}.cpp
#function(add_gtest test)
    #if (TestXMLOutput)
        #set (test_xml_output --gtest_output=xml:${test}.xml)
    #endif(TestXMLOutput)
#
    #if(${ARGC} EQUAL 2)
        #set(source "${ARGV1}/${test}.cpp")
        #set(gtest_src "${ARGV1}/gtest_main.cc;${ARGV1}/gtest-all.cc")
    #else(${ARGC} EQUAL 2)
        #set(source "${test}.cpp")
        #set(gtest_src "gtest/gtest_main.cc;gtest/gtest-all.cc")
    #endif(${ARGC} EQUAL 2)
#
    #add_executable(${test} ${source} ${gtest_src} ${header_files})
    #target_link_libraries(${test} ${LINK_ALL})
    #add_test(NAME ${test} COMMAND ${test} ${test_xml_output})
#endfunction(add_gtest)

function(add_python_test test)
    add_test(NAME ${test} COMMAND ${PYTHON_EXECUTABLE} ${test}.py)
    #add_test(NAME basis_test COMMAND ${PYTHON_EXECUTABLE} basis_test.py)
    set_tests_properties(${test} PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
endfunction(add_python_test)
