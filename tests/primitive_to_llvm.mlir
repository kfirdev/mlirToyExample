// RUN: ../build/src/toy-opt --primitive-to-llvm %s | mlir-translate --mlir-to-llvmir | llc -filetype=obj > %t 
// RUN: clang -c primitive_to_llvm_main.c 
// RUN: clang primitive_to_llvm_main.o %t -o a.out
// RUN: ./a.out | FileCheck %s

// CHECK: 1
func.func @test_primitive_fn(%arg0: !primitive.int<32>) -> !primitive.int<32> {
  %2 = primitive.add %arg0, %arg0 : !primitive.int<32> // 2
  %3 = primitive.mul %2, %2 : !primitive.int<32> // 4
  %4 = primitive.sub %3, %2 : !primitive.int<32> // 2
  %5 = primitive.div %4, %2 : !primitive.int<32> // 1
  return %5 : !primitive.int<32>
}
