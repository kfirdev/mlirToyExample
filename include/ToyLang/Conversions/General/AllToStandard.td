#ifndef LIB_CONVERSION_ALLTOSTANDARD_ALLTOSTANDARD_TD_
#define LIB_CONVERSION_ALLTOSTANDARD_ALLTOSTANDARD_TD_

include "mlir/Pass/PassBase.td"

def AllToStandard : Pass<"toy-to-standard"> {
  let summary = "Lower both `arrays` and `primitive` to standard MLIR dialects.";

  let description = [{
    This pass lowers both `arrays` and `primitive` dialect to standard MLIR
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
    "mlir::tosa::TosaDialect",
    "mlir::toylang::primitive::PrimitiveDialect",
    "mlir::toylang::arrays::ArraysDialect",
	"mlir::func::FuncDialect",
	"mlir::scf::SCFDialect",
  ];
}

#endif
