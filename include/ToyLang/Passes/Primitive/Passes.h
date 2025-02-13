#pragma once

#include "PrintPass.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveOps.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveInterfaces.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <string>

namespace mlir::toylang::primitive{

struct FullUnrollPass : PassWrapper<FullUnrollPass,OperationPass<>>{
	void runOnOperation() override;
	StringRef getArgument() const final { return "full-unroll"; }

  	StringRef getDescription() const final {
  	  return "Fully unroll all loops";
  	}
};

struct HoistConstPass : PassWrapper<HoistConstPass,OperationPass<>>{
	void runOnOperation() override;
	StringRef getArgument() const final { return "hoist-const"; }

  	StringRef getDescription() const final {
  	  return "Hoist constants out of the loop";
  	}
};

namespace passes{

#define GEN_PASS_REGISTRATION 
#include "ToyLang/Passes/Primitive/PrintPass.h.inc"

}

} 
