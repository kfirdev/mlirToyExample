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

//===----------------------------------------------------------------------===//
// Parse IfOp
//===----------------------------------------------------------------------===//
ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result) {
	result.regions.reserve(2);
	Region *thenRegion = result.addRegion();
    Region *elseRegion = result.addRegion();

  	OpAsmParser::UnresolvedOperand cond;
	auto boolType = mlir::toylang::primitive::BoolType::get(parser.getContext());
  	if (parser.parseOperand(cond) ||
  	    parser.resolveOperand(cond, boolType, result.operands))
  	  return failure();

	if (parser.parseOptionalArrowTypeList(result.types))
		return failure();

	if (parser.parseRegion(*thenRegion, /*arguments=*/{}, /*argTypes=*/{}))
		return failure();

	if (!parser.parseOptionalKeyword("else"))
		if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
			return failure();

	if (parser.parseOptionalAttrDict(result.attributes))
		return failure();

	return mlir::success();
}

void IfOp::print(OpAsmPrinter &p) {
  bool printBlockTerminators = false;

  p << " " << getCondition();
  if (!getResults().empty()) {
    p << " -> (" << getResultTypes() << ")";
    // Print yield explicitly if the op defines values.
    printBlockTerminators = true;
  }
  p << ' ';
  p.printRegion(getThenRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/printBlockTerminators);
  if (auto &elseRegion = getElseRegion(); !elseRegion.empty()){
  p << " else ";
  p.printRegion(elseRegion,
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/printBlockTerminators);
  }


  p.printOptionalAttrDict((*this)->getAttrs());
}
//===----------------------------------------------------------------------===//
// Parse ForOp
//===----------------------------------------------------------------------===//
ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::Argument inductionVariable;
  OpAsmParser::UnresolvedOperand lb, ub, step;
  IntegerType type;

  // Parse the induction variable followed by '='.
  if (parser.parseOperand(inductionVariable.ssaName) || parser.parseEqual() ||
      // Parse loop bounds.
      parser.parseOperand(lb) || parser.parseKeyword("to") ||
      parser.parseOperand(ub) || parser.parseKeyword("step") ||
      parser.parseOperand(step))
    return failure();

  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  regionArgs.push_back(inductionVariable);

  bool hasIterArgs = succeeded(parser.parseOptionalKeyword("iter_args"));
  if (hasIterArgs) {
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(result.types))
      return failure();
  }

  if (regionArgs.size() != result.types.size() + 1)
    return parser.emitError(
        parser.getNameLoc(),
        "mismatch in number of loop-carried values and defined values");


  // Parse optional type, else assume Index.
  int width = 0;
  if (parser.parseOptionalColon())
    type = IntegerType::get(parser.getContext(),32);
  else if (parser.parseInteger(width).succeeded())
    type = IntegerType::get(parser.getContext(),width);
  else
    return failure();

  
  // Resolve input operands.
  // Because type is determined later they start with a no type parse it and then resolve
  regionArgs.front().type = type;
  if (parser.resolveOperand(lb, type, result.operands) ||
      parser.resolveOperand(ub, type, result.operands) ||
      parser.resolveOperand(step, type, result.operands))
    return failure();

  if (hasIterArgs) {
    for (auto argOperandType :
         llvm::zip(llvm::drop_begin(regionArgs), operands, result.types)) {
      Type type = std::get<2>(argOperandType);
      std::get<0>(argOperandType).type = type;
      if (parser.resolveOperand(std::get<1>(argOperandType), type,
                                result.operands))
        return failure();
    }
  }
  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();
  //ForOp::ensureTerminator(*body, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

static void printInitializationList(OpAsmPrinter &p,
                                    Block::BlockArgListType blocksArgs,
                                    ValueRange initializers,
                                    StringRef prefix = "") {
  assert(blocksArgs.size() == initializers.size() &&
         "expected same length of arguments and initializers");
  if (initializers.empty())
    return;

  p << prefix << '(';
  llvm::interleaveComma(llvm::zip(blocksArgs, initializers), p, [&](auto it) {
    p << std::get<0>(it) << " = " << std::get<1>(it);
  });
  p << ")";
}


void ForOp::print(OpAsmPrinter &p) {
  p << " " << getInductionVar() << " = " << getLowerBound() << " to "
    << getHigherBound() << " step " << getStep();

  printInitializationList(p, getRegionIterArgs(), getInitArgs(), " iter_args");
  if (!getInitArgs().empty())
    p << " -> (" << getInitArgs().getTypes() << ')';
  p << ' ';
  if (Type t = getInductionVar().getType(); !t.isIndex())
    p << " : " << t << ' ';
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/!getInitArgs().empty());
  p.printOptionalAttrDict((*this)->getAttrs());

}

} // namespace mlir::toylang::primitive
