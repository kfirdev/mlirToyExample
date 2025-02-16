// RUN: ../build/src/toy-opt --toy-to-standard %s > %t
// RUN: FileCheck %s < %t

//CHECK-LABEL: test_dyn_concatOp
func.func @test_dyn_concatOp(%arg0: !arrays.array<0,!primitive.int<32>>, %arg1: !arrays.array<0,!primitive.int<32>>) -> !arrays.array<0,!primitive.int<32>> {
  // CHECK: tensor<?xi32>
  %0 = arrays.concat %arg0, %arg1 : (!arrays.array<0,!primitive.int<32>>, !arrays.array<0,!primitive.int<32>>)
  return %0 : !arrays.array<0,!primitive.int<32>>
}

func.func @test_extractOp_extern(%arg0: !arrays.array<0,!primitive.int<32>>) -> !primitive.int<32> {
  %idx = index.constant 1
  %0 = arrays.extract %arg0[%idx] : !arrays.array<0,!primitive.int<32>>
  return %0 : !primitive.int<32>
}

func.func @test_insertOp_extern(%arg0: !arrays.array<0,!primitive.int<32>>) -> !arrays.array<0,!primitive.int<32>> {
  %idx = index.constant 1
  %value = primitive.constant 1 : 32
  %0 = arrays.insert %value into %arg0[%idx] : !arrays.array<0,!primitive.int<32>>
  return %0 : !arrays.array<0,!primitive.int<32>>
}
