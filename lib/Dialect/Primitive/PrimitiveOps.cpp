#include "include/ToyLang/Dialect/Primitive/PrimitiveOps.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveInterfaces.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/IR/OpImplementation.h"
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
//mlir::OpFoldResult FromStandardOp::fold(FromStandardOp::FoldAdaptor adaptor) {
//  // Returns null if the cast failed, which corresponds to a failed fold.
//  return dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getInput());
//}

}// namespace mlir::toylang::primitive
