#ifndef ARRAYS_DIALECT_TOYDIALECT
#define ARRAYS_DIALECT_TOYDIALECT

include "mlir/IR/DialectBase.td"

// Dialect definition.
def Arrays_Dialect : Dialect {
  let name = "arrays"; // Namespace prefix for the dialect
  let cppNamespace = "::mlir::toylang::arrays"; // C++ namespace

  let description = [{
	The arrays dialect is for defining the arrays built on top of primitive types.
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  //let hasConstantMaterializer = 1;
  let dependentDialects = [
    "mlir::toylang::primitive::PrimitiveDialect",
  ];
}

#endif

