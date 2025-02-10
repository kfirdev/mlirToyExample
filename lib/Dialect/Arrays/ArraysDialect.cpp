#include "include/ToyLang/Dialect/Primitive/PrimitiveDialect.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveOps.h"
#include "include/ToyLang/Dialect/Arrays/ArraysDialect.h"
#include "include/ToyLang/Dialect/Arrays/ArraysType.h"
#include "include/ToyLang/Dialect/Arrays/ArraysAttr.h"
#include "include/ToyLang/Dialect/Arrays/ArraysOps.h"
#include "llvm/ADT/TypeSwitch.h"

#include "include/ToyLang/Dialect/Primitive/PrimitiveTypes.h"
#include "ToyLang/Dialect/Arrays/ArraysDialect.cpp.inc"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"

#define GET_TYPEDEF_CLASSES
#include "ToyLang/Dialect/Arrays/ArraysTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ToyLang/Dialect/Arrays/ArraysAttr.cpp.inc"

#define GET_OP_CLASSES
#include "ToyLang/Dialect/Arrays/ArraysOps.cpp.inc"

namespace mlir {

template<>
struct FieldParser<llvm::SmallVector<mlir::toylang::primitive::IntegerAttr>> {
	static FailureOr<llvm::SmallVector<mlir::toylang::primitive::IntegerAttr>> parse(AsmParser &parser) {
		std::string strValue;
    	if (parser.parseString(&strValue))
    	  return failure();

		llvm::SmallVector<mlir::toylang::primitive::IntegerAttr> array;
		array.reserve(strValue.length());

		auto intType = mlir::toylang::primitive::IntegerType::get(parser.getContext(), 8);

		for (char c : strValue)
			array.push_back(mlir::toylang::primitive::IntegerAttr::get(intType, llvm::APInt{8,static_cast<uint32_t>(c),false,false}));

    	return array;
	}

};

}

namespace llvm{

template <typename T> hash_code hash_value(SmallVector<T> S) {
  return hash_combine_range(S.begin(), S.end());
}

}

namespace mlir::toylang::arrays{

void ArraysDialect::initialize(){
	addTypes<
	#define GET_TYPEDEF_LIST 
	#include "ToyLang/Dialect/Arrays/ArraysTypes.cpp.inc"
		>();

	addOperations<
	#define GET_OP_LIST
	#include "ToyLang/Dialect/Arrays/ArraysOps.cpp.inc"
		>();

	addAttributes<
	#define GET_ATTRDEF_LIST
	#include "ToyLang/Dialect/Arrays/ArraysAttr.cpp.inc"
		>();
}

llvm::LogicalResult ConcatOp::verify(){
	if (getRhs().getType().getLength() + getLhs().getType().getLength() != getResult().getType().getLength()){
		 return emitOpError() << "Result length should have the length of both rhs and lhs combined";
	}
	if (getRhs().getType().getType().getWidth() != getLhs().getType().getType().getWidth() 
			|| getLhs().getType().getType().getWidth() != getResult().getType().getType().getWidth()
			|| getRhs().getType().getType().getWidth() != getResult().getType().getType().getWidth()){
		 return emitOpError() << "all widths must be equal";
	}
	if (getRhs().getType() != getLhs().getType()){
		 return emitOpError() << "all input types must be equal";
	}
	return mlir::success();
}

llvm::LogicalResult ExtractOp::verify(){
	if (getTensor().getType().getType() != getResult().getType()){
		 return emitOpError() << "all types must be equal";
	}
	return mlir::success();
}

::llvm::LogicalResult ConcatOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  if (operands.size() <= 1)
    return ::mlir::failure();

  auto Lhs = mlir::dyn_cast<ArrayType>(operands[0].getType());
  auto Rhs = mlir::dyn_cast<ArrayType>(operands[1].getType());
  ::mlir::Type odsInferredType0 = ArrayType::get(operands[0].getType().getContext(),Lhs.getLength()+Rhs.getLength(),Lhs.getType());

  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor){
	return adaptor.getValue();
}

::mlir::OpFoldResult ConcatOp::fold(ConcatOp::FoldAdaptor adaptor){
	if (adaptor.getRhs() == NULL || adaptor.getLhs() == NULL){
		return nullptr;
	}

	auto lhs = mlir::cast<ArrayAttr>(adaptor.getLhs());
	auto rhs = mlir::cast<ArrayAttr>(adaptor.getRhs());
	llvm::SmallVector<PrimitiveAttrInterface> arr = lhs.getValue();
	arr.append(rhs.getValue());
	return ArrayAttr::get(getContext(),getType(),arr);
}

::mlir::OpFoldResult ExtractOp::fold(ExtractOp::FoldAdaptor adaptor){
	if (adaptor.getIndices() == NULL || adaptor.getTensor() == NULL){
		return nullptr;
	}

	int idx = mlir::cast<mlir::IntegerAttr>(adaptor.getIndices()).getValue().getZExtValue();
	auto tensor = mlir::cast<ArrayAttr>(adaptor.getTensor());
	PrimitiveAttrInterface value = tensor.getValue()[idx];
	return value;
}
::mlir::OpFoldResult InsertOp::fold(InsertOp::FoldAdaptor adaptor){
	if (adaptor.getIndices() == NULL || adaptor.getDest() == NULL || adaptor.getScalar() == NULL){
		return nullptr;
	}

	int idx = mlir::cast<mlir::IntegerAttr>(adaptor.getIndices()).getValue().getZExtValue();
	auto dest = mlir::cast<ArrayAttr>(adaptor.getDest());
	auto scalar = mlir::cast<PrimitiveAttrInterface>(adaptor.getScalar());

	llvm::SmallVector<PrimitiveAttrInterface> newArray = dest.getValue();
	newArray[idx] = scalar;

	return ArrayAttr::get(getContext(),getType(),newArray);
}

mlir::Operation *ArraysDialect::materializeConstant(::mlir::OpBuilder &builder,
                                         ::mlir::Attribute value,
                                         ::mlir::Type type,
                                         ::mlir::Location loc){

	if (auto val = mlir::dyn_cast<ArrayAttr>(value);val){
		return builder.create<ConstantOp>(loc,type,val);
	}
	if (auto val = mlir::dyn_cast<PrimitiveAttrInterface>(value);val){
		return builder.create<primitive::ConstantOp>(loc,type,val);
	}

	return nullptr;

}

} // namespace mlir::toylang::arrays
