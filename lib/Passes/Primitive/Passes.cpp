#include "ToyLang/Passes/Primitive/Passes.h"
#include "ToyLang/Passes/Primitive/PrintPass.h"
#include "ToyLang/Dialect/Primitive/PrimitiveDialect.h"
#include "ToyLang/Dialect/Primitive/PrimitiveAttr.h"
#include "ToyLang/Dialect/Primitive/PrimitiveOps.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveOps.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveInterfaces.h"
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
LogicalResult ForLoopUnroll::matchAndRewrite(ForOp forOp,PatternRewriter &rewriter) const {

  auto lowerConst = forOp.getLowerBound().getDefiningOp<ConstantOp>();
  auto upperConst = forOp.getHigherBound().getDefiningOp<ConstantOp>();
  auto stepConst = forOp.getStep().getDefiningOp<ConstantOp>();
  if (forOp.getLowerBound() == NULL || forOp.getHigherBound() == NULL || forOp.getStep() == NULL){
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
  		if (auto _ = mlir::dyn_cast<YieldOp>(op)){
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

} // namespace
