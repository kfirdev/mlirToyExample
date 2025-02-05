#pragma once

#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::toylang::primitive{

#define GEN_PASS_DECL
#include "ToyLang/Conversions/Primitive/PrimitiveToStandard.h.inc"

#define GEN_PASS_REGISTRATION 
#include "ToyLang/Conversions/Primitive/PrimitiveToStandard.h.inc"

}
