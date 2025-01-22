#pragma once

//#include "mlir/IR/DialectImplementation.h"
#include "include/ToyLang/Dialect/PrimitiveDialect/PrimitiveDialect.h"
#include "include/ToyLang/Dialect/PrimitiveDialect/PrimitiveTypes.h"
#include "mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"

#define GET_OP_CLASSES
#include "ToyLang/Dialect/PrimitiveDialect/PrimitiveOps.h.inc"
