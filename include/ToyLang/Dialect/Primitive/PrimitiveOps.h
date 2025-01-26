#pragma once

//#include "mlir/IR/DialectImplementation.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveDialect.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveTypes.h"
//#include "include/ToyLang/Dialect/Primitive/PrimitiveAttr.h"
#include "mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"

#define GET_OP_CLASSES
#include "ToyLang/Dialect/Primitive/PrimitiveOps.h.inc"
