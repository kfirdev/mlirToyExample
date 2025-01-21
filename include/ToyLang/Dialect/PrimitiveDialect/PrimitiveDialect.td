#ifndef PRIMITIVE_DIALECT_TOYDIALECT
#define PRIMITIVE_DIALECT_TOYDIALECT

include "mlir/IR/DialectBase.td"

// Dialect definition.
def Primitive_Dialect : Dialect {
  let name = "primitive"; // Namespace prefix for the dialect
  let cppNamespace = "::mlir::toylang::primitive"; // C++ namespace

  let description = [{
	The primitive dialect is for defining the primitives types of toylang.
  }];

  let useDefaultTypePrinterParser = 1;
}

#endif

