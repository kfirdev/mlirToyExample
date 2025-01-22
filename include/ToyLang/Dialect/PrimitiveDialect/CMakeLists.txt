set(LLVM_TARGET_DEFINITIONS PrimitiveOps.td)
mlir_tablegen(PrimitiveDialect.h.inc -gen-dialect-decls)
mlir_tablegen(PrimitiveDialect.cpp.inc -gen-dialect-defs)

mlir_tablegen(PrimitiveTypes.h.inc -gen-typedef-decls)
mlir_tablegen(PrimitiveTypes.cpp.inc -gen-typedef-defs)

mlir_tablegen(PrimitiveOps.h.inc -gen-op-decls)
mlir_tablegen(PrimitiveOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRPrimDialectIncGen)
