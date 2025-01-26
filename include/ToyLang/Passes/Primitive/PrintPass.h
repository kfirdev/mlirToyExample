#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::toylang::primitive{
#define GEN_PASS_DECL_PRINTPASS
#include "ToyLang/Passes/Primitive/PrintPass.h.inc"

}
