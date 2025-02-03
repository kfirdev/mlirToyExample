// RUN: ../build/src/toy-opt --canonicalize %s > %t
// RUN: FileCheck %s < %t

// CHECK-LABEL: @test_simple
func.func @test_simple() -> !primitive.int<32> {
  // CHECK: int 3 
  // CHECK-NEXT: return
  %p0 = primitive.constant 2 : 32
  %2 = primitive.add %p0, %p0 : !primitive.int<32> //4
  %3 = primitive.mul %2, %2 : !primitive.int<32> //16
  %4 = primitive.sub %3, %2 : !primitive.int<32> //12
  %5 = primitive.div %4, %2 : !primitive.int<32> //3
  return %5 : !primitive.int<32>
}
// CHECK-LABEL: @test_float
func.func @test_float() -> !primitive.float<32> {
  // CHECK: float 3.8
  // CHECK-NEXT: return
  %p0 = primitive.constant 2.4 : 32
  %2 = primitive.add %p0, %p0 : !primitive.float<32> //4
  %3 = primitive.mul %2, %2 : !primitive.float<32> //16
  %4 = primitive.sub %3, %2 : !primitive.float<32> //12
  %5 = primitive.div %4, %2 : !primitive.float<32> //3
  return %5 : !primitive.float<32>
}

// CHECK-LABEL: @test_bool
func.func @test_bool() -> !primitive.bool {
  // CHECK: bool true
  // CHECK-NEXT: return
  %c_true = primitive.constant true
  %c_false = primitive.constant false
  %0 = primitive.mul %c_true, %c_false : !primitive.bool //false
  %1 = primitive.div %c_true, %0 : !primitive.bool //true
  return %1 : !primitive.bool
}
