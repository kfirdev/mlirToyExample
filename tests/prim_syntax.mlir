// RUN: ../build/src/toy-opt %s > %t
// RUN: FileCheck %s < %t

module {
  func.func @main(%arg0: !primitive.int<10>) -> !primitive.int<10> {
	// CHECK: primitive.int
    return %arg0 : !primitive.int<10>
  }
}

//CHECK-LABEL: test_binops_syntax
func.func @test_binops_syntax(%arg0: !primitive.int<32>,%arg1: !primitive.int<32>) -> !primitive.int<32> {
  // CHECK: primitive.add
  %0 = primitive.add %arg0, %arg1 : !primitive.int<32>
  // CHECK: primitive.mul
  %1 = primitive.mul %0, %0 : !primitive.int<32>
  %2 = primitive.sub %0, %1 : !primitive.int<32>
  %3 = primitive.div %2, %1 : !primitive.int<32>
  return %3 : !primitive.int<32>
}
//CHECK-LABEL: test_constants
func.func @test_constants() -> !primitive.int<10> {

  //CHECK: primitive.constant 
  %0 = primitive.constant -1 : 10 
  //CHECK: primitive.constant 
  %1 = primitive.constant -1.6 : 16 
  %2 = primitive.constant true
  return %0 : !primitive.int<10>
}

//CHECK-LABEL: test_bool 
func.func @test_bool(%arg0: !primitive.bool,%arg1: !primitive.bool) -> !primitive.bool {
  %0 = primitive.mul %arg0, %arg1 : !primitive.bool
  return %0 : !primitive.bool
}

//CHECK-LABEL: test_if
func.func @test_if(%cond: !primitive.bool, %res: !primitive.int<10>) -> !primitive.int<10>{

  // CHECK: primitive.if
  %2 = primitive.if %cond -> !primitive.int<10>{
	// CHECK: primitive.yield
	primitive.yield %res : !primitive.int<10>
  } else {
	%3 = primitive.constant 2 : 10 
	// CHECK: primitive.yield
	primitive.yield %3 : !primitive.int<10>
  }

  return %2 : !primitive.int<10>
}
//CHECK-LABEL: test_for
func.func @test_for() -> !primitive.int<32>{

  %start = primitive.constant 1 : 32
  %end = primitive.constant 4 : 32
  %step = primitive.constant 1 : 32
  %sum_0 = primitive.constant 2 : 32
  %0 = primitive.for %1 = %start to %end step %step 
		iter_args(%sum_iter = %sum_0) -> (!primitive.int<32>){

	  %val = primitive.add %start, %step : !primitive.int<32>

	  primitive.yield %val: !primitive.int<32>
  }

  return %0 : !primitive.int<32>
}
