// RUN: ../build/src/toy-opt --canonicalize %s > %t
// RUN: FileCheck %s < %t

// CHECK-LABEL: @test_simple
func.func @test_simple() -> !primitive.int<32> {
  // CHECK: primitive.constant 8 
  // CHECK-NEXT: return
  %p0 = primitive.constant 2 : !primitive.int<32>
  %2 = primitive.add %p0, %p0 : (!primitive.int<32>, !primitive.int<32>) -> !primitive.int<32>
  %3 = primitive.add %2, %2 : (!primitive.int<32>, !primitive.int<32>) -> !primitive.int<32>
  return %3 : !primitive.int<32>
}

