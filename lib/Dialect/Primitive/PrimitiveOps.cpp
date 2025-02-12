#include "include/ToyLang/Dialect/Primitive/PrimitiveOps.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveInterfaces.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/IRMapping.h"
#include <string>

namespace mlir::toylang::primitive{

mlir::LogicalResult ConstantOp::verify(){

    auto type = getType();
    auto value = getValue();
	
	if (!type || !value)
      return emitOpError("Invalid type for constant");
  
    unsigned bitWidth = type.getWidth();

    if (value.getActiveWidth() > bitWidth) {
		 return emitOpError() << "Value (" << value.getValueStr() << ") exceeds the allowed bit-width (" 
                             << bitWidth << ") of the integer type. The value requires at least "
                             << value.getActiveWidth() << " bits to represent.";
    }
  
    return success();
}

mlir::LogicalResult AddOp::verify(){
	//llvm::errs() << getRhs().getType().getWidth() << "\n";
	if (getResult().getType().hasTrait<IsABool>()) {
		 return emitOpError() << "cannot be applied to type " << getType(); 
	}
	return success();
}
mlir::LogicalResult SubOp::verify(){
	if (getResult().getType().hasTrait<IsABool>()) {
		 return emitOpError() << "cannot be applied to type " << getType(); 
	}
	return success();
}
mlir::LogicalResult DivOp::verify(){
	//if (getResult().getType().hasTrait<IsABool>()) {
	//	 return emitOpError() << "cannot be applied to type " << getType(); 
	//}
	return success();
}
mlir::LogicalResult MultOp::verify(){
	//if (getResult().getType().hasTrait<IsABool>()) {
	//	 return emitOpError() << "cannot be applied to (" << getType() << ")"; 
	//}
	return success();
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type type,PrimitiveAttrInterface value){
  odsState.getOrAddProperties<ConstantOpAdaptor::Properties>().value = value;
  odsState.addTypes(type);
}

mlir::OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor){
	return adaptor.getValue();
}

mlir::OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor){

	if (adaptor.getRhs() == NULL || adaptor.getLhs() == NULL){
		return nullptr;
	}
	auto lhs = mlir::cast<PrimitiveAttrInterface>(adaptor.getOperands()[0]);
	auto rhs = mlir::cast<PrimitiveAttrInterface>(adaptor.getOperands()[1]);
	return lhs.add(rhs);
}

mlir::OpFoldResult SubOp::fold(SubOp::FoldAdaptor adaptor){

	if (adaptor.getRhs() == NULL || adaptor.getLhs() == NULL){
		return nullptr;
	}
	auto lhs = mlir::cast<PrimitiveAttrInterface>(adaptor.getOperands()[0]);
	auto rhs = mlir::cast<PrimitiveAttrInterface>(adaptor.getOperands()[1]);
	return lhs.sub(rhs);
}

mlir::OpFoldResult MultOp::fold(MultOp::FoldAdaptor adaptor){

	if (adaptor.getRhs() == NULL || adaptor.getLhs() == NULL){
		return nullptr;
	}
	auto lhs = mlir::cast<PrimitiveAttrInterface>(adaptor.getOperands()[0]);
	auto rhs = mlir::cast<PrimitiveAttrInterface>(adaptor.getOperands()[1]);
	return lhs.mult(rhs);
}

mlir::OpFoldResult DivOp::fold(DivOp::FoldAdaptor adaptor){

	if (adaptor.getRhs() == NULL || adaptor.getLhs() == NULL){
		return nullptr;
	}
	auto lhs = mlir::cast<PrimitiveAttrInterface>(adaptor.getOperands()[0]);
	auto rhs = mlir::cast<PrimitiveAttrInterface>(adaptor.getOperands()[1]);
	return lhs.div(rhs);
}

llvm::LogicalResult IfOp::fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results){

  if (adaptor.getCondition() == NULL){
	  return failure();
  }

  auto condAttr = mlir::dyn_cast<BoolAttr>(adaptor.getCondition());
  if (!condAttr){
	  return failure();
  }

  bool cond = condAttr.getValue();

  auto& region = cond ? adaptor.getThenRegion() : adaptor.getElseRegion();

  if (region.empty()){
	  return failure();
  }
  if (auto yieldOperands = region.front().getTerminator();yieldOperands){
	  results.push_back(yieldOperands->getOperand(0));
	  return success();
  }

  return failure();
}

struct ForLoopUnroll : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const final {

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
};

void ForOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<ForLoopUnroll>(
      context);
}

Block::BlockArgListType ForOp::getRegionIterArgs() {
  return getRegion().front().getArguments().drop_front(1);
}
SmallVector<Region *> ForOp::getLoopRegions() { return {&getRegion()}; }

LogicalResult ForOp::verify() {
  // Check that the number of init args and op results is the same.
  if (getInitArgs().size() != getNumResults())
    return emitOpError(
        "mismatch in number of loop-carried values and defined values");

  return success();
}

MutableArrayRef<OpOperand> ForOp::getInitsMutable() {
  return getInitArgsMutable();
}


}// namespace mlir::toylang::primitive
