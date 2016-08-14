cmake_minimum_required(VERSION 3.2)
project(captchan)

#set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
 
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
 
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
 
include_directories("${PROJECT_INCLUDE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall")

find_package(ImageMagick REQUIRED COMPONENTS Magick++)

set(GENERATOR_SOURCE_FILES generator/generator.h generator/generator.cc)
add_library(generator SHARED ${GENERATOR_SOURCE_FILES})
target_include_directories(generator PRIVATE ${ImageMagick_INCLUDE_DIRS})
target_link_libraries(generator ${ImageMagick_LIBRARIES})
target_compile_definitions(generator PRIVATE IMAGE_WIDTH=220 IMAGE_HEIGHT=60 FONT_PATH=${CMAKE_SOURCE_DIR}/generator/fonts MAGICKCORE_QUANTUM_DEPTH=16 MAGICKCORE_HDRI_ENABLE=0)

add_executable(generator_test generator/test.cc)
target_link_libraries(generator_test generator)

add_executable(generator_test_vec generator/test_vec.cc)
target_link_libraries(generator_test_vec generator)

add_subdirectory(tiny-dnn)
include_directories(tiny-dnn)

add_executable(learner 
        integration/learner.cc
        tiny-dnn/tiny_cnn/tiny_cnn.h
    )
target_link_libraries(learner generator pthread)

