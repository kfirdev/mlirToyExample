add_mlir_dialect_library(MLIRPrim
	PrimitiveDialect.cpp
	PrimitiveOps.cpp

	DEPENDS
	MLIRPrimDialectIncGen
	#MLIRPrinPassIncGen
	#MLIRPrimDialectTypesIncGen

    )
