#include "include/ToyLang/Dialect/Primitive/PrimitiveOps.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/IR/OpImplementation.h"
#include <string>

namespace mlir::toylang::primitive{

::mlir::ParseResult parseWidth(::mlir::OpAsmParser &parser,unsigned& width){
	if (parser.parseOptionalColon())
  	  return ::mlir::failure();

	if (parser.parseInteger(width)){
		return mlir::failure();
	}
	return mlir::success();
}

::mlir::ParseResult parseIntegerType(::mlir::OpAsmParser &parser,::mlir::Type &outputRawType, llvm::APInt &intValue){

	unsigned width = 0;
	if (parseWidth(parser,width)){
	    width = intValue.getBitWidth();
	}
	
	outputRawType = IntegerType::get(parser.getContext(),width);
	
	if (width < intValue.getActiveBits()){
		std::string valueStr;
		llvm::raw_string_ostream valueStream(valueStr);
		intValue.print(valueStream, true);
		valueStream.flush();
		return parser.emitError(parser.getCurrentLocation()) << "Value (" << valueStr << ") exceeds the allowed bit-width (" 
	                   << width << ") of the integer type. The value requires at least "
	                   << intValue.getActiveBits() << " bits to represent.";
	}
	
	intValue = intValue.sext(width);
	return mlir::success();
}

::mlir::ParseResult parseFloatType(::mlir::OpAsmParser &parser,::mlir::Type &outputRawType,double &floatValue,llvm::APFloat &finalValue){

	unsigned width = 0;
	if (parseWidth(parser,width)){
	    width = 32;
	}
	
	outputRawType = FloatType::get(parser.getContext(),width);

	std::string value = std::to_string(floatValue);
	switch(width){
		case 16:
			finalValue = llvm::APFloat{llvm::APFloat::IEEEhalf(),value};
			break;
		case 32:
			finalValue = llvm::APFloat{llvm::APFloat::IEEEsingle(),value};
			break;
		case 64:
			finalValue = llvm::APFloat{llvm::APFloat::IEEEdouble(),value};
			break;
		default:
			return parser.emitError(parser.getCurrentLocation()) << width 
				<< " is an invalid width for a float choose between 16, 32, 64";

	}
	
	return mlir::success();
}

::mlir::ParseResult parseBoolType(::mlir::OpAsmParser &parser,::mlir::Type &outputRawType){
	outputRawType = BoolType::get(parser.getContext());
	return mlir::success();
}

::mlir::ParseResult parseOptionalBool(::mlir::OpAsmParser &parser,bool &boolValue){
	if (parser.parseOptionalKeyword("true").succeeded()){
		boolValue = true;
		return mlir::success();
	}
	if (parser.parseOptionalKeyword("false").succeeded()){
		boolValue = false;
		return mlir::success();
	}
	return mlir::failure();
}

::mlir::ParseResult parseAttributeAndType(::mlir::OpAsmParser &parser, PrimitiveAttrInterface &attribute,::mlir::Type &outputRawType){

	bool boolValue;
	if (succeeded(parseOptionalBool(parser,boolValue))){
		outputRawType = BoolType::get(parser.getContext());
		attribute = BoolAttr::get(outputRawType,boolValue);
		return mlir::success();
	}

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


	double floatValue;
    if (succeeded(parser.parseFloat(floatValue))){
		if (isNegative){
			floatValue *= -1; 
		}
		llvm::APFloat finalValue{0.};
		if (parseFloatType(parser,outputRawType,floatValue,finalValue))
			return mlir::failure();

		attribute = FloatAttr::get(outputRawType,finalValue);
		return mlir::success();
	}


	return mlir::failure();
}


::mlir::ParseResult ConstantOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  //::mlir::toylang::primitive::IntegerAttr valueAttr;
  PrimitiveAttrInterface valueAttr;
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
	// convert to the interface for the type instead
    if (auto validType = ::llvm::dyn_cast<PrimitiveTypeInterface>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace mlir::toylang::primitive
