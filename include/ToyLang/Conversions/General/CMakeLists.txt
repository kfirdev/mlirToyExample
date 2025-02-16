set(LLVM_TARGET_DEFINITIONS AllToStandard.td)
mlir_tablegen(AllToStandard.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRAllToStanIncGen)

