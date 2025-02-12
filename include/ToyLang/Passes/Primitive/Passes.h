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

struct ForLoopUnroll : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp forOp,PatternRewriter &rewriter) const final;


};
struct FullUnrollPass : PassWrapper<FullUnrollPass,OperationPass<>>{
	void runOnOperation() override;
	StringRef getArgument() const final { return "full-unroll"; }

  	StringRef getDescription() const final {
  	  return "Fully unroll all loops";
  	}
};

namespace passes{

#define GEN_PASS_REGISTRATION 
#include "ToyLang/Passes/Primitive/PrintPass.h.inc"

}

} 
