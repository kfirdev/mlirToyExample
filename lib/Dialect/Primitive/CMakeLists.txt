add_mlir_dialect_library(MLIRPrim
	PrimitiveDialect.cpp
	PrimitiveOps.cpp
	PrimitiveParser.cpp
	PrimitiveInterfaces.cpp

	DEPENDS
	MLIRPrimDialectIncGen
	MLIRPrimInterfaceIncGen
    )
