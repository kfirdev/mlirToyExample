set(LLVM_TARGET_DEFINITIONS ArraysToStandard.td)
mlir_tablegen(ArraysToStandard.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRArrToStanIncGen)
