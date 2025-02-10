#pragma once

#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace mlir::toylang::arrays{

#define GEN_PASS_DECL
#include "ToyLang/Conversions/Arrays/ArraysToStandard.h.inc"

#define GEN_PASS_REGISTRATION 
#include "ToyLang/Conversions/Arrays/ArraysToStandard.h.inc"

}
