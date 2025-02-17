#include "ToyLang/Passes/Primitive/Passes.h"
#include "ToyLang/Passes/Primitive/PrintPass.h"
#include "ToyLang/Dialect/Primitive/PrimitiveDialect.h"
#include "ToyLang/Dialect/Primitive/PrimitiveAttr.h"
#include "ToyLang/Dialect/Primitive/PrimitiveOps.h"
#include "ToyLang/Dialect/Arrays/ArraysOps.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveOps.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveInterfaces.h"
#include "include/ToyLang/Dialect/Arrays/ArraysInterfaces.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"


namespace mlir::toylang::primitive{


#define GEN_PASS_DEF_PRINTPASS
#include "ToyLang/Passes/Primitive/PrintPass.h.inc"
struct PrintPass : impl::PrintPassBase<PrintPass>{
	void runOnOperation() override{
		  getOperation()->walk([&](ConstantOp op) {

		llvm::outs() << "ConstantOp found with bit width: " << op.getValue().getWidth() << "\n";
		llvm::outs() << "Should be with bit width: " << op.getResult().getType().getWidth() << "\n";
	});
	}
};

//===----------------------------------------------------------------------===//
// For loop unroll 
//===----------------------------------------------------------------------===//

struct ForLoopUnroll : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp forOp,PatternRewriter &rewriter) const final{

	auto lowerConst = forOp.getLowerBound().getDefiningOp<ConstantOp>();
  	auto upperConst = forOp.getHigherBound().getDefiningOp<ConstantOp>();
  	auto stepConst = forOp.getStep().getDefiningOp<ConstantOp>();
  	if (!lowerConst || !upperConst || !stepConst){
  	    return failure();
  	}
  	
  	int lower = mlir::cast<IntegerAttr>(lowerConst.getValue()).getValue().getZExtValue();
  	int upper = mlir::cast<IntegerAttr>(upperConst.getValue()).getValue().getZExtValue();
  	int step = mlir::cast<IntegerAttr>(stepConst.getValue()).getValue().getZExtValue();

  	rewriter.setInsertionPointAfter(forOp.getOperation());
  	IRMapping mapping;

  	int i = 0;
  	for (mlir::Value val: forOp.getInitArgs()){
  		mapping.map(forOp.getRegionIterArgs()[i],val);
  		i++;
  	}

  	mlir::Operation* lastCopiedOp;
  	for (int i= lower;i<upper;i+=step){
  		auto constOp = rewriter.create<ConstantOp>(
  	      forOp.getLoc(),forOp.getInductionVar().getType(),
  		  IntegerAttr::get(IntegerType::get(getContext(),32),llvm::APInt{32,(uint64_t)i}));

  		mapping.map(forOp.getInductionVar(),constOp);

  		mlir::Operation* terminatorOp;
  		for (auto& op: forOp.getRegion().front()){
  			if (mlir::dyn_cast<YieldOp>(op)){
  				terminatorOp = rewriter.clone(op,mapping);
  			}
  			else{
  				lastCopiedOp = rewriter.clone(op,mapping);
  			}
  		}

  		int k = 0;
  		if (terminatorOp->getNumOperands() > forOp.getRegionIterArgs().size()){
  			return emitError(terminatorOp->getLoc()) << "more values yielded than there are iter args";
  		}
  		for (auto vals: terminatorOp->getOperands()){
  			mapping.map(forOp.getRegionIterArgs()[k],vals);
  			k++;
  		}
  		rewriter.eraseOp(terminatorOp);
  	}
  	rewriter.replaceOp(forOp.getOperation(),lastCopiedOp);

  	return success();

  }


};

void FullUnrollPass::runOnOperation() {
	mlir::RewritePatternSet patterns(&getContext());
	patterns.add<ForLoopUnroll>(patterns.getContext());
	mlir::GreedyRewriteConfig config;
	config.cseConstants = false;
	config.fold = false;
	config.maxIterations = 1;
	config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;
	config.strictMode = GreedyRewriteStrictness::ExistingOps;
	(void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
}

//===----------------------------------------------------------------------===//
// Hoist constants
//===----------------------------------------------------------------------===//

struct ForHoistConst : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp forOp,PatternRewriter &rewriter) const final{

	SmallVector<Operation*, 4> constOps;
	
	//The crash is happening because you are modifying (replacing) operations 
	//while iterating over the block’s op list directly. When you call rewriter.replaceOp()
	//inside the range-based for loop, you invalidate the iterator for that block
	bool all_const;
    for (Operation &op : forOp.getRegion().front().without_terminator()) {
	  all_const = true;
	  for (auto operand: op.getOperands()){
		  if (operand.getParentRegion() == &forOp.getRegion()){
			  all_const = false; 
		  }
	  }
      if (dyn_cast<ConstantOp>(op) || all_const)
        constOps.push_back(&op);
    }
    // If no constants, nothing to do.
    if (constOps.empty())
      return failure();

    // Hoist each constant out of the loop.
    for (Operation *op : constOps) {
      // Clone the op.
      Operation *cloned = rewriter.clone(*op);
      // Insert the clone before the loop op.
      cloned->moveBefore(forOp);
      // Replace uses of the op inside the loop with the cloned op's results.
      rewriter.replaceOp(op, cloned->getResults());
    }
    return success();
  }
};

void HoistConstPass::runOnOperation() {
	mlir::RewritePatternSet patterns(&getContext());
	patterns.add<ForHoistConst>(patterns.getContext());
	mlir::GreedyRewriteConfig config;
	config.cseConstants = false;
	config.fold = false;
	config.maxIterations = 1;
	config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;
	config.strictMode = GreedyRewriteStrictness::ExistingOps;
	(void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
}

//===----------------------------------------------------------------------===//
// Infer shapes 
//===----------------------------------------------------------------------===//

void ShapeInfrencePass::runOnOperation() {
  FuncOp function = getOperation();
    // Walk all operations in the function.
    function.walk([&](Operation *op) {
      // (1) For cast operations:
      if (auto castOp = dyn_cast<arrays::CastOp>(op)) {
        // Get the source and destination array types.
		if (castOp.getSource().getType().getLength() != 0){
			castOp.replaceAllUsesWith(castOp.getOperand());
			castOp.erase();
		}
		//else if (castOp.getDest().getType().getLength() != 0){
		//	castOp.replaceAllUsesWith(castOp.getResult());
		//}
		//else{
		//	return;
		//}
	  } 
	  else if (auto shapeOp = dyn_cast<arrays::ShapeInference>(op)){
			shapeOp.inferShapes();
	  }

  });
}

} // namespace
