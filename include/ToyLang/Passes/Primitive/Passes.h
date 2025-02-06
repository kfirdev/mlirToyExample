#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::toylang::primitive{

#define GEN_PASS_DECL_PRINTPASS
#include "ToyLang/Passes/Primitive/Passes.h.inc"

#define GEN_PASS_DECL_CONCATREPLACEPASS
#include "ToyLang/Passes/Primitive/Passes.h.inc"

namespace passes{

#define GEN_PASS_REGISTRATION 
#include "ToyLang/Passes/Primitive/Passes.h.inc"

}

} 
