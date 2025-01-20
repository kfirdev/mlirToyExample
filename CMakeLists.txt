cmake_minimum_required(VERSION 3.20.0)

project(mlirToyLang)

# seems like everyone uses 17 with mlir
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# get the environment variable set by the shell.nix
set(LLVM_BUILD_DIR $ENV{LLVM_BUILD_DIR})
# path to llvm
set(LLVM_DIR ${LLVM_BUILD_DIR}/lib/cmake/llvm)
# path to mlir
set(MLIR_DIR ${LLVM_BUILD_DIR}/lib/cmake/mlir)
# path to mlir-tblgen
set(MLIR_TABLEGEN_EXE "${LLVM_BUILD_DIR}/bin/mlir-tblgen")
# build the dependencies so it won't relay on as many exernal files
set(BUILD_DEPS ON)
# Donn't build shared libraries so it wouldn't relay on as many files
set(BUILD_SHARED_LIBS OFF)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

include(AddLLVM)
include(TableGen)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include) #include Tablegen'd files

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(src)

