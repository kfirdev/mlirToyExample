// RUN: ../build/src/toy-opt --infer-shape %s > %t
// RUN: FileCheck %s < %t

//CHECK-LABEL: gen_call
primitive.func @gen_call() -> !arrays.array<6, !primitive.int<32>> {
    %0 = arrays.constant [1,2,3] : !arrays.array<3, !primitive.int<32>>
    %1 = arrays.constant [4,5,6] : !arrays.array<3, !primitive.int<32>>
	//CHECK-NOT: arrays.cast
    %2 = arrays.cast %0 : <3, !primitive.int<32>> to <0, !primitive.int<32>>
    %3 = arrays.cast %1 : <3, !primitive.int<32>> to <0, !primitive.int<32>>
    %4 = arrays.concat %2, %3 : (<0, !primitive.int<32>>, <0, !primitive.int<32>>)
    %5 = arrays.cast %4 : <0, !primitive.int<32>> to <6, !primitive.int<32>>
    primitive.return %5 : !arrays.array<6, !primitive.int<32>>
}

//CHECK-LABEL: test_extractOp
primitive.func @test_extractOp() -> !primitive.int<32> {
  %0 = arrays.constant [1,2,3] : !arrays.array<3, !primitive.int<32>>
  //CHECK-NOT: arrays.cast
  %1 = arrays.cast %0 : <3, !primitive.int<32>> to <0, !primitive.int<32>>
  %idx = index.constant 1
  %2 = arrays.extract %1[%idx] : !arrays.array<0,!primitive.int<32>>
  primitive.return %2 : !primitive.int<32>
}

//CHECK-LABEL: test_insertOp
primitive.func @test_insertOp() -> !arrays.array<3,!primitive.int<32>> {
  %0 = arrays.constant [1,2,3] : !arrays.array<3, !primitive.int<32>>
  //CHECK-NOT: arrays.cast
  %1 = arrays.cast %0 : <3, !primitive.int<32>> to <0, !primitive.int<32>>
  %idx = index.constant 1
  %value = primitive.constant 1 : 32
  %2 = arrays.insert %value into %1[%idx] : !arrays.array<0,!primitive.int<32>>
  %3 = arrays.cast %2 : <0, !primitive.int<32>> to <3, !primitive.int<32>>
  primitive.return %3 : !arrays.array<3,!primitive.int<32>>
}
