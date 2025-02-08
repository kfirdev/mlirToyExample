// RUN: ../build/src/toy-opt %s > %t
// RUN: FileCheck %s < %t


//CHECK-LABEL: test_arrType
func.func @test_arrType(%arg0: !arrays.intArr<10,10>) -> !arrays.intArr<10,10> {
  // CHECK: arrays.int
  return %arg0 : !arrays.intArr<10,10>
}

//CHECK-LABEL: test_constOp
func.func @test_constOp() -> !arrays.intArr<3,10> {
  // CHECK: arrays.constant
  %0 = arrays.constant [1,2,3]  : !arrays.intArr<3,10>
  // CHECK: arrays.constant
  %1 = arrays.constant [false,true,false]  : !arrays.boolArr<3>
  // CHECK: arrays.constant
  %2 = arrays.constant [3.45,4.45,5.45]  : !arrays.floatArr<3,32>
  return %0 : !arrays.intArr<3,10>
}

//CHECK-LABEL: test_concatOp
func.func @test_concatOp(%arg0: !arrays.intArr<3,10>, %arg1: !arrays.intArr<3,10>) -> !arrays.intArr<6,10> {
  %0 = arrays.concat %arg0, %arg1 : (!arrays.intArr<3,10>, !arrays.intArr<3,10>) -> !arrays.intArr<6,10>
  return %0 : !arrays.intArr<6,10>
}
//CHECK-LABEL: test_extractOp
func.func @test_extractOp(%arg0: !arrays.intArr<3,10>) -> !primitive.int<10> {
  %idx = index.constant 1
  %0 = arrays.extract %arg0[%idx] : !arrays.intArr<3,10> -> !primitive.int<10>
  return %0 : !primitive.int<10>
}
