// RUN: ../build/src/toy-opt --hoist-const %s > %t
// RUN: FileCheck %s < %t

//CHECK-LABEL: test_for
func.func @test_for(%addon: !primitive.int<32>) -> !primitive.int<32>{
  %start = primitive.constant 1 : 32
  %end = primitive.constant 4 : 32
  %step = primitive.constant 1 : 32
  %sum_0 = primitive.constant 541 : 32
  // CHECK: primitive.for
  %0 = primitive.for %1 = %addon to %end step %step 
		iter_args(%sum_iter = %sum_0) -> (!primitive.int<32>){
	  %c = primitive.constant 3 : 32
	  %h = primitive.add %c, %start : !primitive.int<32>
	  %l = primitive.add %h, %addon : !primitive.int<32>
	  // CHECK-NEXT: primitive.add
	  %val = primitive.add %sum_iter, %l : !primitive.int<32>
	  // CHECK-NEXT: primitive.yield
	  primitive.yield %val: !primitive.int<32>
  }

  return %0 : !primitive.int<32>
}
