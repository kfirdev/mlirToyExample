// RUN: ../build/src/toy-opt --canonicalize %s > %t
// RUN: FileCheck %s < %t

// CHECK-LABEL: @test_concat
func.func @test_concat() -> !arrays.array<6,!primitive.int<10>> {
  // CHECK: [1,2,3,4,5,6]
  %0 = arrays.constant [1,2,3]  : !arrays.array<3,!primitive.int<10>>
  %1 = arrays.constant [4,5,6]  : !arrays.array<3,!primitive.int<10>>
  %2 = arrays.concat %0, %1 : (!arrays.array<3,!primitive.int<10>>, !arrays.array<3,!primitive.int<10>>)
  return %2 : !arrays.array<6,!primitive.int<10>>
}

// CHECK-LABEL: @test_extract
func.func @test_extract() -> !primitive.int<10> {
  %0 = arrays.constant [1,2,3]  : !arrays.array<3,!primitive.int<10>>
  %idx = index.constant 1
  // CHECK: int 2
  %1 = arrays.extract %0[%idx] : !arrays.array<3,!primitive.int<10>>
  return %1 : !primitive.int<10>
}

// CHECK-LABEL: @test_insert
func.func @test_insert() -> !arrays.array<3,!primitive.int<10>> {
  %0 = arrays.constant [1,2,3]  : !arrays.array<3,!primitive.int<10>>
  %idx = index.constant 1
  %value = primitive.constant 1 : 10
  // CHECK: [1,1,3]
  %1 = arrays.insert %value into %0[%idx] : !arrays.array<3,!primitive.int<10>>
  return %1 : !arrays.array<3,!primitive.int<10>>
}
