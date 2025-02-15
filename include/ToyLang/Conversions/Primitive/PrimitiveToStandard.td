#ifndef LIB_CONVERSION_PRIMTOSTANDARD_PRIMTOSTANDARD_TD_
#define LIB_CONVERSION_PRIMTOSTANDARD_PRIMTOSTANDARD_TD_

include "mlir/Pass/PassBase.td"

def PrimToStandard : Pass<"primitive-to-standard"> {
  let summary = "Lower `primitive` to standard MLIR dialects.";

  let description = [{
    This pass lowers the `primitive` dialect to standard MLIR
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::toylang::primitive::PrimitiveDialect",
  ];
}

#endif
