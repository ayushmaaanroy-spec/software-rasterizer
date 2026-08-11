cmake_minimum_required(VERSION 3.20)
project(software_rasterizer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Debug builds of this are roughly an order of magnitude slower.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

find_package(Threads REQUIRED)

function(sr_set_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
  endif()
endfunction()

# ---------------------------------------------------------------- library
# The pipeline is header-only (it is templated on the shader program), so the
# static library carries the non-templated support code.
add_library(sr STATIC
  src/framebuffer.cpp
  src/mesh.cpp
  src/png.cpp
  src/texture.cpp)

target_include_directories(sr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sr PUBLIC Threads::Threads)
sr_set_warnings(sr)

# ------------------------------------------------------------------- demo
add_executable(srdemo src/main.cpp)
target_link_libraries(srdemo PRIVATE sr)
sr_set_warnings(srdemo)

# ------------------------------------------------------------------ tests
enable_testing()

add_executable(srtests
  tests/check.cpp
  tests/test_math.cpp
  tests/test_pipeline.cpp
  tests/test_png.cpp)
target_link_libraries(srtests PRIVATE sr)
sr_set_warnings(srtests)

add_test(NAME srtests COMMAND srtests)
