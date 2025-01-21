add_mlir_dialect_library(MLIRPrim
	PrimitiveDialect.cpp

	DEPENDS
	MLIRPrimDialectIncGen
	#MLIRPrimDialectTypesIncGen

    )
