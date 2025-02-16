#pragma once

#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::toylang{

#define GEN_PASS_DECL
#include "ToyLang/Conversions/General/AllToStandard.h.inc"

#define GEN_PASS_REGISTRATION 
#include "ToyLang/Conversions/General/AllToStandard.h.inc"

}
