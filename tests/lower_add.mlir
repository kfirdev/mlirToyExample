// RUN: ../build/src/toy-opt --primitive-to-standard %s > %t
// RUN: FileCheck %s < %t


//CHECK-LABEL: test_add_lower
func.func @test_add_lower(%arg0: !primitive.int<10>,%arg1: !primitive.int<10>) -> !primitive.int<10> {
  // CHECK: arith.addi 
  %0 = primitive.add %arg0, %arg1 : !primitive.int<10>
  return %0 : !primitive.int<10>
}
