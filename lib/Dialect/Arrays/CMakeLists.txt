add_mlir_dialect_library(MLIRArr
	ArraysDialect.cpp
	ArraysParser.cpp

	DEPENDS
	MLIRArrDialectIncGen
    )
