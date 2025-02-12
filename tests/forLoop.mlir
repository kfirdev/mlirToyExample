//CHECK-LABEL: test_for
func.func @test_for(%addon: !primitive.int<32>) -> !primitive.int<32>{

  %start = primitive.constant 1 : 32
  %end = primitive.constant 4 : 32
  %step = primitive.constant 1 : 32
  %sum_0 = primitive.constant 541 : 32
  %0 = primitive.for %1 = %start to %end step %step 
		iter_args(%sum_iter = %sum_0) -> (!primitive.int<32>){

	  %val = primitive.add %sum_iter, %step: !primitive.int<32>

	  primitive.yield %val: !primitive.int<32>
  }

  return %0 : !primitive.int<32>
}
