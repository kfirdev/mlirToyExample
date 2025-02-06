#include "ToyLang/Passes/Primitive/Passes.h"
#include "ToyLang/Dialect/Primitive/PrimitiveDialect.h"
#include "ToyLang/Dialect/Primitive/PrimitiveAttr.h"
#include "ToyLang/Dialect/Primitive/PrimitiveOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"


namespace mlir::toylang::primitive{


#define GEN_PASS_DEF_PRINTPASS
#include "ToyLang/Passes/Primitive/Passes.h.inc"

#define GEN_PASS_DEF_CONCATREPLACEPASS
#include "ToyLang/Passes/Primitive/Passes.h.inc"

struct PrintPass : impl::PrintPassBase<PrintPass>{
	void runOnOperation() override{
		  getOperation()->walk([&](ConstantOp op) {

		llvm::outs() << "ConstantOp found with bit width: " << op.getValue().getWidth() << "\n";
		llvm::outs() << "Should be with bit width: " << op.getResult().getType().getWidth() << "\n";
	});
	}
};

struct ConstantReplacePass : impl::ConcatReplacePassBase<ConstantReplacePass>{
	void runOnOperation() override{
		mlir::RewritePatternSet patterns(&getContext());
		tensor::populateDecomposeTensorConcatPatterns(patterns);
		(void)applyPatternsGreedily(getOperation(), std::move(patterns));
	}
};

} // namespace mlir::toylang::primitive
