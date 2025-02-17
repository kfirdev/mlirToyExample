add_mlir_dialect_library(MLIRArr
	ArraysDialect.cpp
	ArraysParser.cpp
	ArraysInterfaces.cpp

	DEPENDS
	MLIRArrDialectIncGen
	MLIRArrInterfaceIncGen
    )
