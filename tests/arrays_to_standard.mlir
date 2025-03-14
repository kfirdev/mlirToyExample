// RUN: ../build/src/toy-opt --toy-to-standard %s > %t
// RUN: FileCheck %s < %t


//CHECK-LABEL: test_constOp
func.func @test_constOp() -> !arrays.array<3,!primitive.int<10>> {
  // CHECK: arith.constant
  %0 = arrays.constant [1,2,3]  : !arrays.array<3,!primitive.int<10>>
  // CHECK: arith.constant
  %1 = arrays.constant [false,true,false]  : !arrays.array<3,!primitive.bool>
  // CHECK: arith.constant
  %2 = arrays.constant [3.45,4.45,5.45]  : !arrays.array<3,!primitive.float<32>>
  return %0 : !arrays.array<3,!primitive.int<10>>
}

//CHECK-LABEL: test_concatOp
func.func @test_concatOp(%arg0: !arrays.array<3,!primitive.int<32>>, %arg1: !arrays.array<3,!primitive.int<32>>) -> !arrays.array<6,!primitive.int<32>> {
  // CHECK: tosa.concat
  %0 = arrays.concat %arg0, %arg1 : (!arrays.array<3,!primitive.int<32>>, !arrays.array<3,!primitive.int<32>>)
  return %0 : !arrays.array<6,!primitive.int<32>>
}
//CHECK-LABEL: test_extractOp
func.func @test_extractOp(%arg0: !arrays.array<3,!primitive.int<10>>) -> !primitive.int<10> {
  %idx = index.constant 1
  // CHECK: tensor.extract
  %0 = arrays.extract %arg0[%idx] : !arrays.array<3,!primitive.int<10>>
  return %0 : !primitive.int<10>
}
//CHECK-LABEL: test_insertOp
func.func @test_insertOp(%arg0: !arrays.array<3,!primitive.int<10>>) -> !arrays.array<3,!primitive.int<10>> {
  %idx = index.constant 1
  %value = primitive.constant 1 : 10
  // CHECK: tensor.insert
  %0 = arrays.insert %value into %arg0[%idx] : !arrays.array<3,!primitive.int<10>>
  return %0 : !arrays.array<3,!primitive.int<10>>
}

