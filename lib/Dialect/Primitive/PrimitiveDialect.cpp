#include "include/ToyLang/Dialect/Primitive/PrimitiveDialect.h"
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

#include "ToyLang/Dialect/Primitive/PrimitiveDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "ToyLang/Dialect/Primitive/PrimitiveTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ToyLang/Dialect/Primitive/PrimitiveAttr.cpp.inc"

#define GET_OP_CLASSES
#include "ToyLang/Dialect/Primitive/PrimitiveOps.cpp.inc"


namespace mlir::toylang::primitive{

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
  auto integerType = mlir::dyn_cast<IntegerType>(type);
  if (value.getBitWidth() != integerType.getWidth()) {
    return Base::get(type.getContext(), type, value.zextOrTrunc(integerType.getWidth()));
  }
  return Base::get(type.getContext(), type, value);

}
llvm::LogicalResult IntegerAttr::verify(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, mlir::Type type, APInt value){
	
	//if (!mlir::isa<mlir::IntegerType>(type)){
	//	return emitError() << "Expected an integer type but got " << type;
	//}
    return success();
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

