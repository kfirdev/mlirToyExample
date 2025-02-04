#ifndef PRIMITIVE_TRAITS_TD_
#define PRIMITIVE_TRAITS_TD_

include "mlir/IR/AttrTypeBase.td"

def IsAnInteger : NativeTypeTrait<"IsAnInteger"> {
  let cppNamespace = "::mlir::toylang::primitive";
}
def IsAFloat : NativeTypeTrait<"IsAFloat"> {
  let cppNamespace = "::mlir::toylang::primitive";
}
def IsABool : NativeTypeTrait<"IsABool"> {
  let cppNamespace = "::mlir::toylang::primitive";
}
def IsAString : NativeTypeTrait<"IsAString"> {
  let cppNamespace = "::mlir::toylang::primitive";
}
#endif
