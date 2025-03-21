#pragma once

#include "include/ToyLang/Dialect/Arrays/ArraysType.h"
#include "include/ToyLang/Dialect/Arrays/ArraysAttr.h"
#include "include/ToyLang/Dialect/Arrays/ArraysInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveInterfaces.h"
#include "mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"

#define GET_OP_CLASSES
#include "ToyLang/Dialect/Arrays/ArraysOps.h.inc"
