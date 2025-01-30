#pragma once

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/APInt.h"
#include <string>

namespace mlir::toylang::primitive{

std::string attrToString(mlir::Attribute attr);

#include "ToyLang/Dialect/Primitive/PrimitiveInterfaces.h.inc"

}
