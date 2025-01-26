set(LLVM_TARGET_DEFINITIONS PrintPass.td)
mlir_tablegen(PrintPass.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRPrinPassIncGen)

