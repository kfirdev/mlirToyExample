#ifndef LIB_PASS_PRINTPASS_PRIMOPS_TD_
#define LIB_PASS_PRINTPASS_PRIMOPS_TD_

include "mlir/Pass/PassBase.td"

def PrintPass : Pass<"print-width"> {
  let summary = "Print the width of constant op";
  let description = [{
	Print the width of the constant op of Primitive Dialect
  }];
  let dependentDialects = ["mlir::toylang::primitive::PrimitiveDialect"];
}

#endif
