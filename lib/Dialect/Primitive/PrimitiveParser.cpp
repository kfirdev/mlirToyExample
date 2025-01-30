#include "include/ToyLang/Dialect/Primitive/PrimitiveOps.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir::toylang::primitive{

::mlir::ParseResult parseIntegerType(::mlir::OpAsmParser &parser,::mlir::Type &outputRawType, llvm::APInt &intValue){
  if (parser.parseColon())
    return ::mlir::failure();
  {
    ::mlir::toylang::primitive::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();

    if (type.getWidth() < intValue.getActiveBits()){
  	std::string valueStr;
  	llvm::raw_string_ostream valueStream(valueStr);
  	intValue.print(valueStream, true);
  	valueStream.flush();
  	return parser.emitError(parser.getCurrentLocation()) << "Value (" << valueStr << ") exceeds the allowed bit-width (" 
                       << type.getWidth() << ") of the integer type. The value requires at least "
                       << intValue.getActiveBits() << " bits to represent.";
      }

    intValue = intValue.sext(type.getWidth());
    outputRawType = type;
  }
  return mlir::success();
}

::mlir::ParseResult parseAttributeAndType(::mlir::OpAsmParser &parser, ::mlir::Attribute &attribute,::mlir::Type &outputRawType){

	bool isNegative = false;
    
    // Check for a leading '-' (negative sign)
    if (succeeded(parser.parseOptionalMinus())) {
        isNegative = true;
    }

	llvm::APInt intValue;
	if (parser.parseOptionalInteger(intValue).has_value()){
		if (isNegative){
			intValue.negate();
		}
		if (parseIntegerType(parser,outputRawType,intValue))
			return mlir::failure();
		attribute = IntegerAttr::get(outputRawType,intValue);
		return mlir::success();
	}

	llvm::APFloat floatValue{0.};
    if (succeeded(parser.parseFloat(llvm::APFloat::IEEEdouble(),floatValue))){
		if (isNegative){
			floatValue.changeSign();
		}
		llvm::errs() << "Value: ";
		floatValue.print(llvm::errs());
		llvm::errs() << "\n";
        return parser.emitError(parser.getCurrentLocation(), "floats currently not supported");
	}


	return mlir::failure();
}


::mlir::ParseResult ConstantOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  //::mlir::toylang::primitive::IntegerAttr valueAttr;
  mlir::Attribute valueAttr;
  ::mlir::Type outputRawType{};
  ::llvm::ArrayRef<::mlir::Type> outputTypes(&outputRawType, 1);

  if (parseAttributeAndType(parser,valueAttr,outputRawType)){
	  return ::mlir::failure();
  }
  
  if (valueAttr) result.getOrAddProperties<ConstantOp::Properties>().value = valueAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  result.addTypes(outputTypes);
  return ::mlir::success();
}

void ConstantOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getValueAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("value");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOutput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::toylang::primitive::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace mlir::toylang::primitive
