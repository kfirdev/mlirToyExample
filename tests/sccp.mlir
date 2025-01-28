// RUN: ../build/src/toy-opt --sccp %s > %t
// RUN: FileCheck %s < %t

// CHECK-LABEL: @test_simple
func.func @test_simple() -> !primitive.int<10> {
  // CHECK: primitive.constant 4 
  // CHECK-NEXT: primitive.constant 2
  // CHECK-NEXT: return
  %p0 = primitive.constant 2 : !primitive.int<10>
  %2 = primitive.add %p0, %p0 : !primitive.int<10>
  return %2 : !primitive.int<10>
}

