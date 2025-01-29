#include "include/ToyLang/Dialect/Primitive/PrimitiveOps.h"

namespace mlir::toylang::primitive{

mlir::LogicalResult ConstantOp::verify(){
    auto type = mlir::dyn_cast<IntegerType>(getType());
    auto value = getValue().getValue();
  
    if (!type)
      return emitOpError("Invalid type for constant");
  
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


void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type type,IntegerAttr value){
  //auto val = IntegerAttr::get(type,value.getValue());
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

}
