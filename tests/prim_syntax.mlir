// RUN: ../build/src/toy-opt %s > %t
// RUN: FileCheck %s < %t

module {
  func.func @main(%arg0: !primitive.int<10>) -> !primitive.int<10> {
	// CHECK: primitive.int
    return %arg0 : !primitive.int<10>
  }
}

//CHECK-LABEL: test_add_syntax
func.func @test_add_syntax(%arg0: !primitive.int<10>,%arg1: !primitive.int<10>) -> !primitive.int<10> {
  // CHECK: primitive.add
  %0 = primitive.add %arg0, %arg1 : !primitive.int<10>
  return %0 : !primitive.int<10>
}
//CHECK-LABEL: test_constants
func.func @test_constants() -> !primitive.int<4> {

  //CHECK: primitive.constant 
  %0 = primitive.constant -1 : !primitive.int<4>
  return %0 : !primitive.int<4>
}
