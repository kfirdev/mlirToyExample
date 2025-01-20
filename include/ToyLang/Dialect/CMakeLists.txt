set(LLVM_TARGET_DEFINITIONS ToyDialect.td)
mlir_tablegen(ToyDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ToyDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRDialectIncGen)
