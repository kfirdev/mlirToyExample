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

// CHECK-LABEL: @test_bool
func.func @test_bool(%arg0: !primitive.bool) -> !primitive.bool {
  // CHECK: arith.andi
  %0 = primitive.mul %arg0, %arg0: !primitive.bool
  // CHECK: arith.ori
  %1 = primitive.div %0, %arg0 : !primitive.bool
  return %1 : !primitive.bool
}
// CHECK-LABEL: @test_string
func.func @test_string(%arg0: !primitive.string) -> !primitive.string {
  // CHECK: arith.constant
  // CHECK-NEXT: tensor.cast
  %0 = primitive.constant "hello "
  // CHECK: arith.constant
  // CHECK-NEXT: tensor.cast
  %1 = primitive.constant "world"
  // CHECK: arith.constant
  // CHECK-NEXT: tensor.cast
  %2 = primitive.add %0, %1 : !primitive.string
  // CHECK: tensor.concat
  %3 = primitive.add %0, %arg0 : !primitive.string
  return %2: !primitive.string
}
