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

//CHECK-LABEL: test_string 
func.func @test_string(%arg0: !primitive.string,%arg1: !primitive.string) -> !primitive.string {
  %p0 = primitive.constant "hello world"
  %0 = primitive.add %arg0, %arg1 : !primitive.string
  return %0 : !primitive.string
}
