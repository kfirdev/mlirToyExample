#pragma once

#include "mlir/IR/DialectImplementation.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveDialect.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveAttr.h"
//#include "include/ToyLang/Dialect/Primitive/PrimitiveInterfaces.h"
//#include "mlir/IR/BuiltinOps.h"    // from @llvm-project
//#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
//#include "mlir/IR/Dialect.h"

#define GET_ATTRDEF_CLASSES
#include "ToyLang/Dialect/Arrays/ArraysAttr.h.inc"
