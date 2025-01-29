// RUN: ../build/src/toy-opt --primitive-to-llvm %s | mlir-translate --mlir-to-llvmir | llc -filetype=obj > %t 
// RUN: clang -c primitive_to_llvm_main.c 
// RUN: clang primitive_to_llvm_main.o %t -o a.out
// RUN: ./a.out | FileCheck %s

// CHECK: 12
func.func @test_primitive_fn(%arg0: !primitive.int<32>) -> !primitive.int<32> {
  %c1 = primitive.constant 5 : !primitive.int<32>
  %0 = primitive.add %arg0, %c1 : !primitive.int<32>
  %1 = primitive.add %0, %0 : !primitive.int<32>
  return %1 : !primitive.int<32>
}
