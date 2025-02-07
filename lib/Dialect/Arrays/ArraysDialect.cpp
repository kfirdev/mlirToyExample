#include "include/ToyLang/Dialect/Primitive/PrimitiveDialect.h"
#include "include/ToyLang/Dialect/Arrays/ArraysDialect.h"
#include "include/ToyLang/Dialect/Arrays/ArraysAttr.h"
#include "include/ToyLang/Dialect/Arrays/ArraysType.h"
#include "include/ToyLang/Dialect/Arrays/ArraysOps.h"
#include "llvm/ADT/TypeSwitch.h"

#include "include/ToyLang/Dialect/Primitive/PrimitiveTypes.h"
#include "ToyLang/Dialect/Arrays/ArraysDialect.cpp.inc"
#include "llvm/ADT/SmallVector.h"

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

} // namespace mlir::toylang::arrays
