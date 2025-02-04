// RUN: ../build/src/toy-opt %s > %t
// RUN: FileCheck %s < %t

//CHECK-LABEL: test_string 
func.func @test_string() -> !primitive.string {
  %p0 = primitive.constant "hello world"
  return %p0 : !primitive.string
}
