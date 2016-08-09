cmake_minimum_required(VERSION 3.2)
project(captchan)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall -Werror")

find_package(ImageMagick REQUIRED COMPONENTS Magick++)

set(GENERATOR_SOURCE_FILES generator/generator.cc)
add_library(generator SHARED ${GENERATOR_SOURCE_FILES})
target_include_directories(generator PRIVATE ${ImageMagick_INCLUDE_DIRS})
target_link_libraries(generator ${ImageMagick_LIBRARIES})
target_compile_definitions(generator PRIVATE IMAGE_WIDTH=220 IMAGE_HEIGHT=60 FONT_PATH=${CMAKE_SOURCE_DIR}/generator/fonts MAGICKCORE_QUANTUM_DEPTH=16 MAGICKCORE_HDRI_ENABLE=0)

#add_executable(captchan captchan.cc)
