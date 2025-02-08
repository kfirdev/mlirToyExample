#pragma once

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/APInt.h"
#include "mlir/Transforms/DialectConversion.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveInterfaces.h"
#include <string>

namespace mlir::toylang::arrays{
	using primitive::PrimitiveTypeInterface;

#include "ToyLang/Dialect/Arrays/ArraysAttrInterfaces.h.inc"
#include "ToyLang/Dialect/Arrays/ArraysTypeInterfaces.h.inc"

}
