#pragma once

#include "mlir/IR/DialectImplementation.h"
#include "include/ToyLang/Dialect/Arrays/ArraysInterface.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveInterfaces.h"

namespace mlir::toylang::arrays{
	using primitive::PrimitiveTypeInterface;
}

#define GET_TYPEDEF_CLASSES 
#include "ToyLang/Dialect/Arrays/ArraysTypes.h.inc"


