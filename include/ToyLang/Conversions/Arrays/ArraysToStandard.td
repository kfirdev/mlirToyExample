#ifndef LIB_CONVERSION_ARRTOSTANDARD_ARRTOSTANDARD_TD_
#define LIB_CONVERSION_ARRTOSTANDARD_ARRTOSTANDARD_TD_

include "mlir/Pass/PassBase.td"

def ArrToStandard : Pass<"arrays-to-standard"> {
  let summary = "Lower `arrays` to standard MLIR dialects.";

  let description = [{
    This pass lowers the `arrays` dialect to standard MLIR
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
    "mlir::toylang::primitive::PrimitiveDialect",
    "mlir::toylang::arrays::ArraysDialect",
  ];
}

#endif
