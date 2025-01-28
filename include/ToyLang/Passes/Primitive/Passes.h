#pragma once

#include "PrintPass.h"

namespace mlir::toylang::primitive{
namespace passes{

#define GEN_PASS_REGISTRATION 
#include "ToyLang/Passes/Primitive/PrintPass.h.inc"

}

} 
