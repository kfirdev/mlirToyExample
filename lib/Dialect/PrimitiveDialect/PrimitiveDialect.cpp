#include "include/ToyLang/Dialect/PrimitiveDialect/PrimitiveDialect.h"

#include "include/ToyLang/Dialect/PrimitiveDialect/PrimitiveTypes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ToyLang/Dialect/PrimitiveDialect/PrimitiveDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "ToyLang/Dialect/PrimitiveDialect/PrimitiveTypes.cpp.inc"


namespace mlir::toylang::primitive{

void PrimitiveDialect::initialize(){
	addTypes<
#define GET_TYPEDEF_LIST 
#include "ToyLang/Dialect/PrimitiveDialect/PrimitiveTypes.cpp.inc"
		>();
}

}

