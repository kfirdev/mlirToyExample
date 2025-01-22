// RUN: ../build/src/toy-opt %s
//CHECK-LABEL: test_add_syntax

module {
  func.func @main(%arg0: !primitive.int<10>,%arg1: !primitive.int<10>) -> !primitive.int<10> {
    %0 = primitive.add %arg0, %arg1 : (!primitive.int<10>, !primitive.int<10>) -> !primitive.int<10>
    return %0 : !primitive.int<10>
  }
}
