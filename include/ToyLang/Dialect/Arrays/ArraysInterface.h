#pragma once

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/APInt.h"
#include "mlir/Transforms/DialectConversion.h"
#include <string>

namespace mlir::toylang::arrays{

#include "ToyLang/Dialect/Arrays/ArraysAttrInterfaces.h.inc"
#include "ToyLang/Dialect/Arrays/ArraysTypeInterfaces.h.inc"

}
