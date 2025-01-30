set(LLVM_TARGET_DEFINITIONS PrimitiveOps.td)
mlir_tablegen(PrimitiveDialect.h.inc -gen-dialect-decls)
mlir_tablegen(PrimitiveDialect.cpp.inc -gen-dialect-defs)

mlir_tablegen(PrimitiveTypes.h.inc -gen-typedef-decls)
mlir_tablegen(PrimitiveTypes.cpp.inc -gen-typedef-defs)

mlir_tablegen(PrimitiveOps.h.inc -gen-op-decls)
mlir_tablegen(PrimitiveOps.cpp.inc -gen-op-defs)

mlir_tablegen(PrimitiveAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(PrimitiveAttr.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRPrimDialectIncGen)

set(LLVM_TARGET_DEFINITIONS PrimitiveInterfaces.td)
mlir_tablegen(PrimitiveInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(PrimitiveInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRPrimInterfaceIncGen)
