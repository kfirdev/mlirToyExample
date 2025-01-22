#include "include/ToyLang/Dialect/PrimitiveDialect/PrimitiveDialect.h"

#include "include/ToyLang/Dialect/PrimitiveDialect/PrimitiveTypes.h"
#include "include/ToyLang/Dialect/PrimitiveDialect/PrimitiveOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ToyLang/Dialect/PrimitiveDialect/PrimitiveDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "ToyLang/Dialect/PrimitiveDialect/PrimitiveTypes.cpp.inc"

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
}

}

