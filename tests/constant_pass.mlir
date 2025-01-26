// RUN: ../build/src/toy-opt --print-width %s > %t
// RUN: FileCheck %s < %t

// CHECK-LABEL: @test_simple
func.func @test_simple(){
  %0 = primitive.constant 2 : !primitive.int<32>
  %1 = primitive.constant 2 : !primitive.int<10>
  %2 = primitive.constant 2 : !primitive.int<4>
  return
}

