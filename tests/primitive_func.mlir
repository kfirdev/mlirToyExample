// RUN: ../build/src/toy-opt %s > %t
// RUN: FileCheck %s < %t

//CHECK-LABEL: test_if
primitive.func @test_if(%cond: !primitive.bool, %res: !primitive.int<10>) -> !primitive.int<10>{

  // CHECK: primitive.if
  %2 = primitive.if %cond -> !primitive.int<10>{
	// CHECK: primitive.yield
	primitive.yield %res : !primitive.int<10>
  } else {
	%3 = primitive.constant 2 : 10 
	// CHECK: primitive.yield
	primitive.yield %3 : !primitive.int<10>
  }

  primitive.return %2 : !primitive.int<10>
}
//CHECK-LABEL: test_for
primitive.func @test_for() -> !primitive.int<32>{

  %start = primitive.constant 1 : 32
  %end = primitive.constant 4 : 32
  %step = primitive.constant 1 : 32
  %sum_0 = primitive.constant 2 : 32
  %0 = primitive.for %1 = %start to %end step %step 
		iter_args(%sum_iter = %sum_0) -> (!primitive.int<32>){

	  %val = primitive.add %start, %step : !primitive.int<32>

	  primitive.yield %val: !primitive.int<32>
  }

  primitive.return %0 : !primitive.int<32>
}
