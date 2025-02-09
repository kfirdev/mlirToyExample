#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::toylang::arrays{

#define GEN_PASS_DECL_CONCATREPLACEPASS
#include "ToyLang/Passes/Arrays/Passes.h.inc"

namespace passes{

#define GEN_PASS_REGISTRATION 
#include "ToyLang/Passes/Arrays/Passes.h.inc"

}

} 
