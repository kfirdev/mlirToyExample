#include "include/ToyLang/Dialect/Primitive/PrimitiveDialect.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveInterfaces.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveAttr.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveTypes.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include <string>

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
template <>
struct FieldParser<llvm::APFloat> {
	static FailureOr<llvm::APFloat> parse(AsmParser &parser) {
		llvm::APFloat value{0.0};
    	if (parser.parseFloat(llvm::APFloat::IEEEdouble(),value))
    	  return failure();
    	return value;
	}

};

}

#include "ToyLang/Dialect/Primitive/PrimitiveDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "ToyLang/Dialect/Primitive/PrimitiveTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ToyLang/Dialect/Primitive/PrimitiveAttr.cpp.inc"

#define GET_OP_CLASSES
#include "ToyLang/Dialect/Primitive/PrimitiveOps.cpp.inc"


namespace mlir::toylang::primitive{
#include "ToyLang/Dialect/Primitive/PrimitiveAttrInterfaces.cpp.inc"
#include "ToyLang/Dialect/Primitive/PrimitiveTypeInterfaces.cpp.inc"

void PrimitiveDialect::initialize(){
	addTypes<
#define GET_TYPEDEF_LIST 
#include "ToyLang/Dialect/Primitive/PrimitiveTypes.cpp.inc"
		>();

	addOperations<
	#define GET_OP_LIST
	#include "ToyLang/Dialect/Primitive/PrimitiveOps.cpp.inc"
		>();

	addAttributes<
		#define GET_ATTRDEF_LIST
		#include "ToyLang/Dialect/Primitive/PrimitiveAttr.cpp.inc"
		>();

}

IntegerAttr IntegerAttr::get(Type type, const APInt &value) {
  return Base::get(type.getContext(), type, value);
}

FloatAttr FloatAttr::get(Type type, const APFloat &value) {
  return Base::get(type.getContext(), type, value);
}
FloatAttr FloatAttr::get(mlir::MLIRContext* context,Type type, const APFloat &value) {
  return Base::get(context, type, value);
}

BoolAttr BoolAttr::get(Type type, bool value){
	return Base::get(type.getContext(),type,value);
}
BoolAttr BoolAttr::get(mlir::MLIRContext* context,Type type, bool value){
	return Base::get(context,type,value);
}

llvm::LogicalResult IntegerAttr::verify(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, mlir::Type type, APInt value){
	// These should NEVER fail if this fails something is wrong with the initialization of the attribute.
	//IntegerType intType = mlir::dyn_cast<IntegerType>(type);
	//if (!intType){
	//	return failure();
	//}
	//if (intType.getWidth() != value.getBitWidth()){
	//	 return emitError() << "Wrong bit width";
	//}
    return success();
}

mlir::Operation *PrimitiveDialect::materializeConstant(::mlir::OpBuilder &builder,
                                         ::mlir::Attribute value,
                                         ::mlir::Type type,
                                         ::mlir::Location loc){
	auto val = mlir::dyn_cast<PrimitiveAttrInterface>(value);
	if (!val)
		return nullptr;

	return builder.create<ConstantOp>(loc,type,val);
}

} // namespace mlir::toylang::primitive

