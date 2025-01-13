cmake_minimum_required(VERSION 3.13)
project(MLIRExample)

set(CC clang)
set(CXX clang++)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_EXPORT_COMPILE_COMMANDS True)

# Find the LLVM package.
find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

# Find the MLIR package.
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

# Define the executable.
add_executable(mlir_example src/main.cpp)

# Link against MLIR and LLVM libraries.
target_link_libraries(mlir_example
  PRIVATE
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  LLVMCore
  LLVMSupport
)

# Include directories for MLIR and LLVM.
target_include_directories(mlir_example
  PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

# Define necessary compile options.
target_compile_options(mlir_example
  PRIVATE
  ${LLVM_COMPILE_FLAGS}
)

