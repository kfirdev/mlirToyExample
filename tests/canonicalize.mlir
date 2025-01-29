// RUN: ../build/src/toy-opt --canonicalize %s > %t
// RUN: FileCheck %s < %t

// CHECK-LABEL: @test_simple
func.func @test_simple() -> !primitive.int<32> {
  // CHECK: primitive.constant 3 
  // CHECK-NEXT: return
  %p0 = primitive.constant 2 : !primitive.int<32>
  %2 = primitive.add %p0, %p0 : !primitive.int<32> //4
  %3 = primitive.mul %2, %2 : !primitive.int<32> //16
  %4 = primitive.sub %3, %2 : !primitive.int<32> //12
  %5 = primitive.div %4, %2 : !primitive.int<32> //3
  return %5 : !primitive.int<32>
}

