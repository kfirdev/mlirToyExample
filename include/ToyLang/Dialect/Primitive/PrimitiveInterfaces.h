#pragma once

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/APInt.h"
#include "mlir/Transforms/DialectConversion.h"
#include <string>
//#include "include/ToyLang/Dialect/Primitive/PrimitiveOps.h"

namespace mlir::toylang::primitive{

#include "ToyLang/Dialect/Primitive/PrimitiveAttrInterfaces.h.inc"
#include "ToyLang/Dialect/Primitive/PrimitiveTypeInterfaces.h.inc"

}
