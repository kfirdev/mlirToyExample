// RUN: ../build/src/toy-opt --primitive-to-standard %s > %t
// RUN: FileCheck %s < %t

// CHECK-LABEL: @test_simple
func.func @test_simple(%arg0: !primitive.int<32>) -> !primitive.int<32> {
  // CHECK: arith.addi
  %2 = primitive.add %arg0, %arg0 : !primitive.int<32>
  // CHECK: arith.muli
  %3 = primitive.mul %2, %2 : !primitive.int<32>
  // CHECK: arith.subi
  %4 = primitive.sub %3, %2 : !primitive.int<32>
  // CHECK: arith.divsi
  %5 = primitive.div %4, %2 : !primitive.int<32>
  return %5 : !primitive.int<32>
}

// CHECK-LABEL: @test_simple_float
func.func @test_simple_float(%arg0: !primitive.float<32>) -> !primitive.float<32> {
  // CHECK: arith.addf
  %2 = primitive.add %arg0, %arg0 : !primitive.float<32>
  // CHECK: arith.mulf
  %3 = primitive.mul %2, %2 : !primitive.float<32>
  // CHECK: arith.subf
  %4 = primitive.sub %3, %2 : !primitive.float<32>
  // CHECK: arith.divf
  %5 = primitive.div %4, %2 : !primitive.float<32>
  return %5 : !primitive.float<32>
}

