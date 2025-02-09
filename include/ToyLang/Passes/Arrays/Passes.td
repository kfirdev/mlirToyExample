#ifndef ARRAYS_PASSES_LIBS 
#define ARRAYS_PASSES_LIBS

include "mlir/Pass/PassBase.td"

def ConcatReplacePass : Pass<"replace-concat"> {
  let summary = "replace concat";
  let description = [{
  replace concat
  }];
  let dependentDialects = ["mlir::tensor::TensorDialect"];
}

#endif
