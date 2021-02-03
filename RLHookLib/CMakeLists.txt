cmake_minimum_required(VERSION 3.10)
project(RLHookLib)

set(CMAKE_CXX_STANDARD 17)

find_package(Git QUIET)
if(GIT_FOUND AND (NOT EXISTS "${PROJECT_SOURCE_DIR}/libs/pybind11/CMakeLists.txt" OR NOT EXISTS "${PROJECT_SOURCE_DIR}/libs/Detours/Detours/README.md"))
    message(STATUS "Cloning pybind11 and detours")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive)
endif()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/libs/pybind11/CMakeLists.txt" OR NOT EXISTS "${PROJECT_SOURCE_DIR}/libs/Detours/Detours/README.md")
    message(FATAL_ERROR "The submodules pybind11 and/or Detours have not been cloned! Clone them with \"git submodule update --init --recursive\".")
endif()

add_subdirectory(libs)
find_package(OpenGL REQUIRED)

add_subdirectory(RLHookDLL)
if(NOT DEFINED ONLY_ADDITIONAL_BINARIES)
    add_subdirectory(PyRLHook)
endif()
add_subdirectory(FunctionAddressGetter)
add_subdirectory(Utils)
