set(LLVM_TARGET_DEFINITIONS ArraysOps.td)
mlir_tablegen(ArraysDialect.h.inc -gen-dialect-decls -dialect arrays)
mlir_tablegen(ArraysDialect.cpp.inc -gen-dialect-defs -dialect arrays)

mlir_tablegen(ArraysTypes.h.inc -gen-typedef-decls --typedefs-dialect arrays)
mlir_tablegen(ArraysTypes.cpp.inc -gen-typedef-defs --typedefs-dialect arrays)

mlir_tablegen(ArraysOps.h.inc -gen-op-decls -dialect arrays)
mlir_tablegen(ArraysOps.cpp.inc -gen-op-defs -dialect arrays)

mlir_tablegen(ArraysAttr.h.inc -gen-attrdef-decls --attrdefs-dialect arrays)
mlir_tablegen(ArraysAttr.cpp.inc -gen-attrdef-defs --attrdefs-dialect arrays)
add_public_tablegen_target(MLIRArrDialectIncGen)

set(LLVM_TARGET_DEFINITIONS ArraysInterface.td)
#mlir_tablegen(ArraysAttrInterfaces.h.inc -gen-attr-interface-decls)
#mlir_tablegen(ArraysAttrInterfaces.cpp.inc -gen-attr-interface-defs)
#
#mlir_tablegen(ArraysTypeInterfaces.h.inc -gen-type-interface-decls)
#mlir_tablegen(ArraysTypeInterfaces.cpp.inc -gen-type-interface-defs)

mlir_tablegen(ArraysOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ArraysOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRArrInterfaceIncGen)
