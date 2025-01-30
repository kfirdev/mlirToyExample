add_mlir_dialect_library(MLIRPrim
	PrimitiveDialect.cpp
	PrimitiveOps.cpp
	PrimitiveParser.cpp

	DEPENDS
	MLIRPrimDialectIncGen
	#MLIRPrinPassIncGen
	#MLIRPrimDialectTypesIncGen

    )
