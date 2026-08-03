cmake_minimum_required(VERSION 3.15)
project(pyodin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# pybind11 from pip or conda: pass -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
# (build.sh does this automatically).
find_package(pybind11 CONFIG REQUIRED)

# Vendored non-ROS Odin SDK. Override with -DODIN_SDK_DIR=/abs/path if needed.
set(ODIN_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../odin_sdk" CACHE PATH "Odin SDK dir")
include_directories(${ODIN_SDK_DIR}/include)

execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
if(ARCH STREQUAL "aarch64" OR ARCH STREQUAL "arm64")
  set(ODIN_LIB ${ODIN_SDK_DIR}/lib/liblydHostApi_arm.a)
else()
  set(ODIN_LIB ${ODIN_SDK_DIR}/lib/liblydHostApi_amd.a)
endif()
message(STATUS "pyodin: arch=${ARCH}  sdk_lib=${ODIN_LIB}")

pybind11_add_module(pyodin pyodin_module.cpp)
# Static archive first, then its dependencies (libusb, pthread, and the glibc
# split-out libs dl/rt/m) so the linker can resolve the archive's undefined
# symbols. dl/rt/m are no-ops on glibc >= 2.34 where they are folded into libc.
target_link_libraries(pyodin PRIVATE ${ODIN_LIB} usb-1.0 pthread ${CMAKE_DL_LIBS} rt m)
