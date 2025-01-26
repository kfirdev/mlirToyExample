#include "include/ToyLang/Dialect/PrimitiveDialect/PrimitiveDialect.h"
#include "include/ToyLang/Dialect/PrimitiveDialect/PrimitiveAttr.h"
#include "include/ToyLang/Dialect/PrimitiveDialect/PrimitiveTypes.h"
#include "include/ToyLang/Dialect/PrimitiveDialect/PrimitiveOps.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iostream>

namespace mlir {
template <>
struct FieldParser<llvm::APInt> {
	static FailureOr<llvm::APInt> parse(AsmParser &parser) {
		llvm::APInt value;
    	if (parser.parseInteger(value))
    	  return failure();
    	return value;
	}

};

}

#include "ToyLang/Dialect/PrimitiveDialect/PrimitiveDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "ToyLang/Dialect/PrimitiveDialect/PrimitiveTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ToyLang/Dialect/PrimitiveDialect/PrimitiveAttr.cpp.inc"

#define GET_OP_CLASSES
#include "ToyLang/Dialect/PrimitiveDialect/PrimitiveOps.cpp.inc"


namespace mlir::toylang::primitive{

void PrimitiveDialect::initialize(){
	addTypes<
#define GET_TYPEDEF_LIST 
#include "ToyLang/Dialect/PrimitiveDialect/PrimitiveTypes.cpp.inc"
		>();

	addOperations<
	#define GET_OP_LIST
	#include "ToyLang/Dialect/PrimitiveDialect/PrimitiveOps.cpp.inc"
		>();

	addAttributes<
		#define GET_ATTRDEF_LIST
		#include "ToyLang/Dialect/PrimitiveDialect/PrimitiveAttr.cpp.inc"
		>();
}

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

llvm::LogicalResult IntegerAttr::verify(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, mlir::Type type, APInt value){
	
	//if (!mlir::isa<mlir::IntegerType>(type)){
	//	return emitError() << "Expected an integer type but got " << type;
	//}
    return success();
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type type,IntegerAttr value){
  odsState.getOrAddProperties<ConstantOpAdaptor::Properties>().value = value;
  odsState.addTypes(type);
}


IntegerAttr IntegerAttr::get(Type type, const APInt &value) {
  auto integerType = mlir::dyn_cast<IntegerType>(type);
  if (value.getBitWidth() != integerType.getWidth()) {
    return Base::get(type.getContext(), type, value.zextOrTrunc(integerType.getWidth()));
  }
  return Base::get(type.getContext(), type, value);

}


mlir::OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor){
	return adaptor.getValue();
}

mlir::OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor){
	auto lhs = mlir::cast<IntegerAttr>(adaptor.getLhs()).getValue();
	auto rhs = mlir::cast<IntegerAttr>(adaptor.getRhs()).getValue();
	auto result = IntegerAttr::get(getType(),lhs+rhs);
	return result;
}

mlir::Operation *PrimitiveDialect::materializeConstant(::mlir::OpBuilder &builder,
                                         ::mlir::Attribute value,
                                         ::mlir::Type type,
                                         ::mlir::Location loc){
	auto val = mlir::dyn_cast<IntegerAttr>(value);

	if (!val)
		return nullptr;

	return builder.create<ConstantOp>(loc,type,val);
}


}

