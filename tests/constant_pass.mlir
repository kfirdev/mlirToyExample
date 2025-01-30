// RUN: ../build/src/toy-opt --print-width %s > %t
// RUN: FileCheck %s < %t

// CHECK-LABEL: @test_simple
func.func @test_simple(){
  %0 = primitive.constant 2 : 32
  %1 = primitive.constant 2 : 10
  %2 = primitive.constant 2 : 4
  return
}

