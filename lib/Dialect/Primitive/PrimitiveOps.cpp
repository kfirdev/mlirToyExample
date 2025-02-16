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

void ForOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<ForLoopUnroll,ForHoistConst>(
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


void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

llvm::LogicalResult ReturnOp::verify() {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>((*this)->getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << results.size() << ")";

  auto inputType = *operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType || llvm::isa<mlir::UnrankedTensorType>(inputType) ||
      llvm::isa<mlir::UnrankedTensorType>(resultType))
    return mlir::success();

  return emitError() << "type of return operand (" << inputType
                     << ") doesn't match function result type (" << resultType
                     << ")";
}

LogicalResult GenericCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;


  if (getResult().getType() != fnType.getResult(0)) {
     auto diag = emitOpError("result type mismatch at index ") << 0;
     diag.attachNote() << "      op result types: " << getResult().getType();
     diag.attachNote() << "function result types: " << fnType.getResults();
     return diag;
  }

  return success();
}

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Set the callee for the generic call operation, this is required by the call
/// interface.
void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", mlir::cast<SymbolRefAttr>(callee));
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }

/// Get the argument operands to the called function as a mutable range, this is
/// required by the call interface.
MutableOperandRange GenericCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}



}// namespace mlir::toylang::primitive
