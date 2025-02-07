// RUN: ../build/src/toy-opt %s > %t
// RUN: FileCheck %s < %t


//CHECK-LABEL: test_arrType
func.func @test_arrType(%arg0: !arrays.intArr<10,10>) -> !arrays.intArr<10,10> {
  // CHECK: arrays.int
  return %arg0 : !arrays.intArr<10,10>
}

//CHECK-LABEL: test_constOp
func.func @test_constOp() -> !arrays.intArr<3,10> {
  // CHECK: arrays.constant [1,2,3]
  %0 = arrays.constant [1,2,3]  : !arrays.intArr<3,10>
  %1 = arrays.constant [false,true,false]  : !arrays.boolArr<3>
  %2 = arrays.constant [3.4561,4.4561,5.4561]  : !arrays.floatArr<3,32>
  return %0 : !arrays.intArr<3,10>
}
