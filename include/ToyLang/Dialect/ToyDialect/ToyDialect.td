#ifndef TOYLANG_DIALECT_TOYDIALECT
#define TOYLANG_DIALECT_TOYDIALECT

include "mlir/IR/DialectBase.td"

// Dialect definition.
def ToyDialect : Dialect {
  let name = "toy"; // Namespace prefix for the dialect
  let cppNamespace = "::mlir::toylang::toy"; // C++ namespace

  let description = [{
    The ToyLang dialect provides operations and constructs for the ToyLang
    programming language. This dialect is designed to be beginner-friendly
    while producing highly optimized binaries.
  }];
}

#endif
