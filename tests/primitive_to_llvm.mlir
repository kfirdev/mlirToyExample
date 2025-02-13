// RUN: ../build/src/toy-opt --primitive-to-llvm %s | mlir-translate --mlir-to-llvmir | llc -filetype=obj > %t 
// RUN: clang -c primitive_to_llvm_main.c 
// RUN: clang primitive_to_llvm_main.o %t -o a.out
// RUN: ./a.out | FileCheck %s

// CHECK: 1
func.func @test_primitive_fn(%arg0: !primitive.int<32>) -> !primitive.int<32> {
  %2 = primitive.add %arg0, %arg0 : !primitive.int<32> // 2
  %3 = primitive.mul %2, %2 : !primitive.int<32> // 4
  %4 = primitive.sub %3, %2 : !primitive.int<32> // 2
  %5 = primitive.div %4, %2 : !primitive.int<32> // 1
  return %5 : !primitive.int<32>
}
// CHECK: 4.28
func.func @test_primitive_fn_double(%arg0: !primitive.float<32>) -> !primitive.float<32> {
  %2 = primitive.add %arg0, %arg0 : !primitive.float<32> // 2.4
  %3 = primitive.mul %2, %2 : !primitive.float<32> // 5.76
  %4 = primitive.sub %3, %2 : !primitive.float<32> // 3.36
  %5 = primitive.div %4, %2 : !primitive.float<32> // 1.4
  return %5 : !primitive.float<32>
}
//CHECK: true
func.func @test_bool(%arg0: !primitive.bool, %arg1: !primitive.bool) -> !primitive.bool { //arg0 = false arg1 = true
  %0 = primitive.mul %arg0, %arg1: !primitive.bool // false
  %1 = primitive.div %0, %arg1 : !primitive.bool // true
  return %1 : !primitive.bool
}

// CHECK: 10
func.func @test_if(%cond: !primitive.bool, %res: !primitive.int<32>) -> !primitive.int<32>{ //cond = true, res = 10
  %2 = primitive.if %cond -> !primitive.int<32>{
	primitive.yield %res : !primitive.int<32>
  } else {
	%3 = primitive.constant 2 : 32 
	primitive.yield %3 : !primitive.int<32>
  }
  return %2 : !primitive.int<32>
}

// CHECK: 9
func.func @test_for(%start: !primitive.int<32>,%end: !primitive.int<32>) -> !primitive.int<32>{ // start = 1, end = 10 
  %step = primitive.constant 1 : 32
  %sum_0 = primitive.constant 0 : 32
  %0 = primitive.for %1 = %start to %end step %step 
		iter_args(%sum_iter = %sum_0) -> (!primitive.int<32>){
	  %val = primitive.add %sum_iter, %step: !primitive.int<32>
	  primitive.yield %val: !primitive.int<32>
  }

  return %0 : !primitive.int<32>
}
