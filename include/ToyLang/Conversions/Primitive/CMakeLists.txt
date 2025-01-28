set(LLVM_TARGET_DEFINITIONS PrimitiveToStandard.td)
mlir_tablegen(PrimitiveToStandard.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRPrinToStanIncGen)

