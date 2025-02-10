// RUN: ../build/src/toy-opt --arrays-to-standard %s > %t
// RUN: FileCheck %s < %t

//CHECK-LABEL: test_dyn_concatOp
func.func @test_dyn_concatOp(%arg0: !arrays.array<0,!primitive.int<32>>, %arg1: !arrays.array<0,!primitive.int<32>>) -> !arrays.array<0,!primitive.int<32>> {
  // CHECK: tensor<?xi32>
  %0 = arrays.concat %arg0, %arg1 : (!arrays.array<0,!primitive.int<32>>, !arrays.array<0,!primitive.int<32>>)
  return %0 : !arrays.array<0,!primitive.int<32>>
}
