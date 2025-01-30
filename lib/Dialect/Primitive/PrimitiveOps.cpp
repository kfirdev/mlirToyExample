#include "include/ToyLang/Dialect/Primitive/PrimitiveOps.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir::toylang::primitive{

mlir::LogicalResult ConstantOp::verify(){
	// Probably better to have an interface for these to not directly convert to integer type because it might not be.

    if (!getType().hasTrait<IsAnInteger>())
      return emitOpError("Invalid type for constant");

	// should be casted to the type interface when it is finished.
    auto type = mlir::dyn_cast<IntegerType>(getType());
	// should be catsted to the attribute interface when it is finished which will directly return a string.
    auto value = mlir::cast<IntegerAttr>(getValue()).getValue();
  
  
    unsigned bitWidth = type.getWidth();

    if (value.getActiveBits() > bitWidth) {
		std::string valueStr;
        llvm::raw_string_ostream valueStream(valueStr);
        value.print(valueStream, true);
        valueStream.flush();

		 return emitOpError() << "Value (" << valueStr << ") exceeds the allowed bit-width (" 
                             << bitWidth << ") of the integer type. The value requires at least "
                             << value.getActiveBits() << " bits to represent.";
    }
  
    return success();
}


void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type type,Attribute value){
  odsState.getOrAddProperties<ConstantOpAdaptor::Properties>().value = value;
  odsState.addTypes(type);
}

mlir::OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor){
	// Need to understand if this becomes a problem...
	return adaptor.getValue();
}

// ops will done or from the interface itself or in some other way.
// currently unknown...
mlir::OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor){

	if (adaptor.getRhs() == NULL || adaptor.getLhs() == NULL){
		return nullptr;
	}
	auto lhs = mlir::cast<IntegerAttr>(adaptor.getOperands()[0]).getValue();
	auto rhs = mlir::cast<IntegerAttr>(adaptor.getOperands()[1]).getValue();
	auto result = IntegerAttr::get(getType(),lhs+rhs);
	return result;
}

mlir::OpFoldResult SubOp::fold(SubOp::FoldAdaptor adaptor){

	if (adaptor.getRhs() == NULL || adaptor.getLhs() == NULL){
		return nullptr;
	}
	auto lhs = mlir::cast<IntegerAttr>(adaptor.getOperands()[0]).getValue();
	auto rhs = mlir::cast<IntegerAttr>(adaptor.getOperands()[1]).getValue();
	auto result = IntegerAttr::get(getType(),lhs-rhs);
	return result;
}

mlir::OpFoldResult MultOp::fold(MultOp::FoldAdaptor adaptor){

	if (adaptor.getRhs() == NULL || adaptor.getLhs() == NULL){
		return nullptr;
	}
	auto lhs = mlir::cast<IntegerAttr>(adaptor.getOperands()[0]).getValue();
	auto rhs = mlir::cast<IntegerAttr>(adaptor.getOperands()[1]).getValue();
	auto result = IntegerAttr::get(getType(),lhs*rhs);
	return result;
}

mlir::OpFoldResult DivOp::fold(DivOp::FoldAdaptor adaptor){

	if (adaptor.getRhs() == NULL || adaptor.getLhs() == NULL){
		return nullptr;
	}
	auto lhs = mlir::cast<IntegerAttr>(adaptor.getOperands()[0]).getValue();
	auto rhs = mlir::cast<IntegerAttr>(adaptor.getOperands()[1]).getValue();
	auto result = IntegerAttr::get(getType(),lhs.sdiv(rhs));
	return result;
}

}// namespace mlir::toylang::primitive
