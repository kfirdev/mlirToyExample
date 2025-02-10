// RUN: ../build/src/toy-opt --arrays-to-llvm %s | mlir-translate --mlir-to-llvmir | llc -filetype=obj > %t 
// RUN: clang++ -c arrays_to_llvm_main.cpp
// RUN: clang++ arrays_to_llvm_main.o %t -L/home/kfirby/personal/cpp_projects/llvm_project/llvm-project/build/lib/ -lmlir_c_runner_utils -o b.out
// RUN: ./b.out | FileCheck %s

// CHECK: [1,2,3,4,5,6]
func.func @test_concatOp_extern(%arg0: !arrays.array<3,!primitive.int<32>>, %arg1: !arrays.array<3,!primitive.int<32>>) -> !arrays.array<6,!primitive.int<32>> {
  %0 = arrays.concat %arg0, %arg1 : (!arrays.array<3,!primitive.int<32>>, !arrays.array<3,!primitive.int<32>>)
  return %0 : !arrays.array<6,!primitive.int<32>>
}

// CHECK: [1,2,3,4,5,6]
func.func @test_dyn_concatOp_extern(%arg0: !arrays.array<0,!primitive.int<32>>, %arg1: !arrays.array<0,!primitive.int<32>>) -> !arrays.array<0,!primitive.int<32>> {
  %0 = arrays.concat %arg0, %arg1 : (!arrays.array<0,!primitive.int<32>>, !arrays.array<0,!primitive.int<32>>)
  return %0 : !arrays.array<0,!primitive.int<32>>
}

// CHECK: 2
func.func @test_extractOp_extern(%arg0: !arrays.array<3,!primitive.int<32>>) -> !primitive.int<32> {
  %idx = index.constant 1
  %0 = arrays.extract %arg0[%idx] : !arrays.array<3,!primitive.int<32>>
  return %0 : !primitive.int<32>
}

// CHECK: [1,1,3] 
func.func @test_insertOp_extern(%arg0: !arrays.array<3,!primitive.int<32>>) -> !arrays.array<3,!primitive.int<32>> {
  %idx = index.constant 1
  %value = primitive.constant 1 : 32
  %0 = arrays.insert %value into %arg0[%idx] : !arrays.array<3,!primitive.int<32>>
  return %0 : !arrays.array<3,!primitive.int<32>>
}

