#include "ToyLang/Passes/Arrays/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"


namespace mlir::toylang::arrays{


#define GEN_PASS_DEF_CONCATREPLACEPASS
#include "ToyLang/Passes/Arrays/Passes.h.inc"

struct ConstantReplacePass : impl::ConcatReplacePassBase<ConstantReplacePass>{
	void runOnOperation() override{
		mlir::RewritePatternSet patterns(&getContext());
		tensor::populateDecomposeTensorConcatPatterns(patterns);
		(void)applyPatternsGreedily(getOperation(), std::move(patterns));
	}
};

} // namespace
