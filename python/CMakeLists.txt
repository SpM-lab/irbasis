# Copy files into a build directories
configure_file(${CMAKE_SOURCE_DIR}/python/irbasis.py ${CMAKE_BINARY_DIR}/irbasis/irbasis.py COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/python/__init__.py ${CMAKE_BINARY_DIR}/irbasis/__init__.py COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/version ${CMAKE_BINARY_DIR}/irbasis/version COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/python/setup.py ${CMAKE_BINARY_DIR}/setup.py COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/python/setup.cfg ${CMAKE_BINARY_DIR}/setup.cfg COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/database/irbasis.h5 ${CMAKE_BINARY_DIR}/irbasis/irbasis.h5 COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/README.md ${CMAKE_BINARY_DIR}/README.md COPYONLY)
