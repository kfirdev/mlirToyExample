// RUN: ../build/src/toy-opt %s

module {
  func.func @main(%arg0: !primitive.int<10>) -> !primitive.int<10> {
    return %arg0 : !primitive.int<10>
  }
}
