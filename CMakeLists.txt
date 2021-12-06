cmake_minimum_required(VERSION 2.8...3.20)
project(DIP)

IF (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF ()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ")

find_package(OpenCV 3)
if (NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if (NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif ()
endif ()

MESSAGE("OPENCV VERSION:${OpenCV_VERSION}")

find_package(Eigen3 3.1.0 REQUIRED)

# check if we have our python libs files
# sudo apt-get install python-matplotlib python-numpy python2.7-dev
find_package(PythonLibs 2.7)
if(PYTHONLIBS_FOUND AND NOT DISABLE_MATPLOTLIB)
    add_definitions(-DHAVE_PYTHONLIBS=1)
    message(STATUS "PYTHON VERSION: " ${PYTHONLIBS_VERSION_STRING})
    message(STATUS "PYTHON INCLUDE: " ${PYTHON_INCLUDE_DIRS})
    message(STATUS "PYTHON LIBRARIES: " ${PYTHON_LIBRARIES})
endif()

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${PYTHON_INCLUDE_DIRS}
        )
# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${PYTHON_LIBRARIES}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        )
##################################################
# Make the core library
##################################################
add_library(${PROJECT_NAME} SHARED
        src/lab1.cpp
        include/lab1.h
        src/lab2.cpp
        include/lab2.h src/lab3.cpp
        include/lab3.h
        thirdparty/matplotlib-cpp/matplotlibcpp.h)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Assigment/)

#Lab1
add_executable(lab1_main
        Assigment/lab1_main.cpp)
target_link_libraries(lab1_main
        ${PROJECT_NAME}
        ${thirdparty_libraries})
#Lab2
add_executable(lab2_main
        Assigment/lab2_main.cpp)
target_link_libraries(lab2_main
        ${PROJECT_NAME}
        ${thirdparty_libraries})
#Lab3
add_executable(lab3_main
        Assigment/lab3_main.cpp)
target_link_libraries(lab3_main
        ${PROJECT_NAME}
        ${thirdparty_libraries})