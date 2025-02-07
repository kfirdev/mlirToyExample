#include "include/ToyLang/Dialect/Arrays/ArraysDialect.h"
#include "include/ToyLang/Dialect/Arrays/ArraysOps.h"
#include "include/ToyLang/Dialect/Arrays/ArraysAttr.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveDialect.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveTypes.h"
#include "llvm/ADT/APInt.h"
#include <cstdlib>
#include <vector>

namespace mlir::toylang::arrays{
//===----------------------------------------------------------------------===//
// Parse Int Array
//===----------------------------------------------------------------------===//
::mlir::ParseResult parseOptionalIntArrayVals(::mlir::OpAsmParser &parser, llvm::SmallVector<llvm::APInt> &arr, unsigned &max_width){
	llvm::APInt result;
	bool has_next = true;
	while (parser.parseOptionalInteger(result).has_value() && has_next){
		if (result.getActiveBits() > max_width){
			max_width = result.getActiveBits();
		}
		arr.push_back(result);
		if(parser.parseOptionalComma().failed()){
			has_next = false;
		}
	}
	if (arr.size() == 0){
		return mlir::failure();
	}
	return mlir::success();
}

::mlir::ParseResult parseIntTypeAndAttr(
		::mlir::OpAsmParser &parser, ::mlir::Attribute &attr, 
		::mlir::Type &outputRawType, llvm::SmallVector<llvm::APInt> &arr, unsigned max_width){
	if(parser.parseRSquare()){
		return mlir::failure();
	}

	if (parser.parseColon())
		return ::mlir::failure();

	unsigned width;
	unsigned length;
	{
		IntegerArrType type;
		if (parser.parseCustomTypeWithFallback(type))
			return ::mlir::failure();
		width = type.getWidth();
		length = type.getLength();
		outputRawType = type;
	}

	if (length != arr.size()){
        return parser.emitError(parser.getCurrentLocation()) << "length should be: " << length;
	}
	if(max_width > width){
		return parser.emitError(parser.getCurrentLocation()) << "bit width ( " << width 
			<< " ) is too small use a bit width of at least: " << max_width;
	}
	llvm::SmallVector<primitive::IntegerAttr> final_array;
	final_array.reserve(arr.size());
	auto tp = primitive::IntegerType::get(parser.getContext(),width);
	for (llvm::APInt val: arr){
		final_array.push_back(std::move(primitive::IntegerAttr::get(tp,val.sext(width))));
	}

	attr = IntegerArrAttr::get(parser.getContext(),outputRawType,final_array);
	return mlir::success();
}
//===----------------------------------------------------------------------===//
// Parse Float Array
//===----------------------------------------------------------------------===//
::mlir::ParseResult parseFloatArrayVals(::mlir::OpAsmParser &parser, llvm::SmallVector<double> &arr){
	double result;
	bool has_next = true;
	while (has_next && parser.parseFloat(result).succeeded()){
		arr.push_back(result);
		if(parser.parseOptionalComma().failed()){
			has_next = false;
		}
	}
	if (arr.size() == 0){
		return mlir::failure();
	}
	return mlir::success();
}

llvm::APFloat parseFloatValue(double &floatValue,unsigned width){

	std::string value = std::to_string(floatValue);
	switch(width){
		case 16:
			return llvm::APFloat{llvm::APFloat::IEEEhalf(),value};
		case 32:
			return llvm::APFloat{llvm::APFloat::IEEEsingle(),value};
		case 64:
			return llvm::APFloat{llvm::APFloat::IEEEdouble(),value};
		default:
			return llvm::APFloat{llvm::APFloat::IEEEdouble(),0};
	}
	
}
::mlir::ParseResult parseFloatTypeAndAttr(
		::mlir::OpAsmParser &parser, ::mlir::Attribute &attr, 
		::mlir::Type &outputRawType, llvm::SmallVector<double> &arr){
	if(parser.parseRSquare()){
		return mlir::failure();
	}

	if (parser.parseColon())
		return ::mlir::failure();

	unsigned width;
	unsigned length;
	{
		FloatArrType type;
		if (parser.parseCustomTypeWithFallback(type))
			return ::mlir::failure();
		width = type.getWidth();
		length = type.getLength();
		outputRawType = type;
	}

	if (length != arr.size()){
        return parser.emitError(parser.getCurrentLocation()) << "length should be: " << length;
	}
	if (width != 16 && width != 32 && width != 64){
        return parser.emitError(parser.getCurrentLocation()) << "can't use this width chose between 16,32,64";
	}
	llvm::SmallVector<primitive::FloatAttr> final_array;
	final_array.reserve(arr.size());
	auto tp = primitive::FloatType::get(parser.getContext(),width);
	for (double val: arr){
		final_array.push_back(std::move(primitive::FloatAttr::get(tp,parseFloatValue(val,width))));
	}

	attr = FloatArrAttr::get(parser.getContext(),outputRawType,final_array);
	return mlir::success();
}
//===----------------------------------------------------------------------===//
// Parse Bool Array
//===----------------------------------------------------------------------===//
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
::mlir::ParseResult parseOptionalBoolArrayVals(::mlir::OpAsmParser &parser, llvm::SmallVector<bool> &arr){
	bool result;
	bool has_next = true;
	while (parseOptionalBool(parser,result).succeeded() && has_next){
		arr.push_back(result);
		if(parser.parseOptionalComma().failed()){
			has_next = false;
		}
	}
	if (arr.size() == 0){
		return mlir::failure();
	}
	return mlir::success();
}

::mlir::ParseResult parseBoolTypeAndAttr(
		::mlir::OpAsmParser &parser, ::mlir::Attribute &attr, 
		::mlir::Type &outputRawType, llvm::SmallVector<bool> &arr){
	if(parser.parseRSquare()){
		return mlir::failure();
	}

	if (parser.parseColon())
		return ::mlir::failure();

	unsigned length;
	{
		BoolArrType type;
		if (parser.parseCustomTypeWithFallback(type))
			return ::mlir::failure();
		length = type.getLength();
		outputRawType = type;
	}

	if (length != arr.size()){
        return parser.emitError(parser.getCurrentLocation()) << "length should be: " << length;
	}
	llvm::SmallVector<primitive::BoolAttr> final_array;
	final_array.reserve(arr.size());
	auto tp = primitive::BoolType::get(parser.getContext());
	for (bool val: arr){
		final_array.push_back(std::move(primitive::BoolAttr::get(tp,val)));
	}

	attr = BoolArrAttr::get(parser.getContext(),outputRawType,final_array);
	return mlir::success();
}

//===----------------------------------------------------------------------===//
// Parse Array
//===----------------------------------------------------------------------===//
::mlir::ParseResult parseOptionalArray(::mlir::OpAsmParser &parser, ::mlir::Attribute &attr, ::mlir::Type &outputRawType){
	if(parser.parseLSquare()){
		return mlir::failure();
	}

	llvm::SmallVector<bool> arrBool;
	if (parseOptionalBoolArrayVals(parser,arrBool).succeeded()){
		if(parseBoolTypeAndAttr(parser,attr,outputRawType,arrBool).succeeded()){
			return mlir::success();
		}
		return mlir::failure();
	}

	bool first_neg = false;
	if (parser.parseOptionalMinus().succeeded()){
		first_neg = true;
	}

	unsigned max_width = 0;
	llvm::SmallVector<llvm::APInt> arrInt;
	if (parseOptionalIntArrayVals(parser,arrInt,max_width).succeeded()){
		if (first_neg){
			arrInt[0].negate();
		}
		if(parseIntTypeAndAttr(parser,attr,outputRawType,arrInt,max_width).succeeded()){
			return mlir::success();
		}
		return mlir::failure();
	}

	llvm::SmallVector<double> arrFloat;
	if (parseFloatArrayVals(parser,arrFloat).succeeded()){
		if (first_neg){
		arrFloat[0] *= -1;
		}
		if(parseFloatTypeAndAttr(parser,attr,outputRawType,arrFloat).succeeded()){
			return mlir::success();
		}
		return mlir::failure();
	}

	return mlir::success();
}
//===----------------------------------------------------------------------===//
// ConstantOp parsing
//===----------------------------------------------------------------------===//
::mlir::ParseResult ConstantOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  mlir::Attribute valueAttr;
  mlir::Type outputRawType{};
  ::llvm::ArrayRef<::mlir::Type> outputTypes(&outputRawType, 1);

  if (parseOptionalArray(parser,valueAttr,outputRawType)){
    return ::mlir::failure();
  }


  //if (parser.parseCustomAttributeWithFallback(valueAttr, ::mlir::Type{})) {
    //return ::mlir::failure();
  //}
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
  //if (parser.parseColon())
  //  return ::mlir::failure();

  //if (parser.parseType(outputRawType))
  //  return ::mlir::failure();
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
   _odsPrinter << getOutput().getType();
}

//===----------------------------------------------------------------------===//
// IntegerArrAttr parsing 
//===----------------------------------------------------------------------===//

::mlir::Attribute IntegerArrAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::Type> _result_type;

  if (odsType) {
    if (auto reqType = ::llvm::dyn_cast<::mlir::Type>(odsType)) {
      _result_type = reqType;
    } else {
      odsParser.emitError(odsLoc, "invalid kind of type specified");
      return {};
    }
  }
  ::mlir::FailureOr<llvm::SmallVector<primitive::IntegerAttr>> _result_value;

  // Parse variable 'value'
  _result_value = ::mlir::FieldParser<llvm::SmallVector<primitive::IntegerAttr>>::parse(odsParser);
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Arr_IntegerArrAttr parameter 'value' which is to be a `llvm::SmallVector<primitive::IntegerAttr>`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return IntegerArrAttr::get(odsParser.getContext(),
      ::mlir::Type((_result_type.value_or(::mlir::NoneType::get(odsParser.getContext())))),
      llvm::SmallVector<primitive::IntegerAttr>((*_result_value)));
}

void IntegerArrAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << '[';
  for (int i = 0; i < getValue().size(); ++i){
	  odsPrinter << getValue()[i].getValueStr() ;
	  if (i < getValue().size()-1){
		odsPrinter << ',';
	  }
  }
  odsPrinter << ']';
  //odsPrinter.printStrippedAttrOrType(getValue());
}

//===----------------------------------------------------------------------===//
// FloatArrAttr parsing
//===----------------------------------------------------------------------===//

::mlir::Attribute FloatArrAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::Type> _result_type;

  if (odsType) {
    if (auto reqType = ::llvm::dyn_cast<::mlir::Type>(odsType)) {
      _result_type = reqType;
    } else {
      odsParser.emitError(odsLoc, "invalid kind of type specified");
      return {};
    }
  }
  ::mlir::FailureOr<llvm::SmallVector<primitive::FloatAttr>> _result_value;

  // Parse variable 'value'
  _result_value = ::mlir::FieldParser<llvm::SmallVector<primitive::FloatAttr>>::parse(odsParser);
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Arr_FloatArrAttr parameter 'value' which is to be a `llvm::SmallVector<primitive::FloatAttr>`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return FloatArrAttr::get(odsParser.getContext(),
      ::mlir::Type((_result_type.value_or(::mlir::NoneType::get(odsParser.getContext())))),
      llvm::SmallVector<primitive::FloatAttr>((*_result_value)));
}

void FloatArrAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << '[';
  for (int i = 0; i < getValue().size(); ++i){
	  odsPrinter << getValue()[i].getValueStr() ;
	  if (i < getValue().size()-1){
		odsPrinter << ',';
	  }
  }
  odsPrinter << ']';
  //odsPrinter.printStrippedAttrOrType(getValue());
}

//===----------------------------------------------------------------------===//
// BoolArrAttr parsing 
//===----------------------------------------------------------------------===//

::mlir::Attribute BoolArrAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::Type> _result_type;

  if (odsType) {
    if (auto reqType = ::llvm::dyn_cast<::mlir::Type>(odsType)) {
      _result_type = reqType;
    } else {
      odsParser.emitError(odsLoc, "invalid kind of type specified");
      return {};
    }
  }
  ::mlir::FailureOr<llvm::SmallVector<primitive::BoolAttr>> _result_value;

  // Parse variable 'value'
  _result_value = ::mlir::FieldParser<llvm::SmallVector<primitive::BoolAttr>>::parse(odsParser);
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Arr_BoolArrAttr parameter 'value' which is to be a `llvm::SmallVector<primitive::BoolAttr>`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return BoolArrAttr::get(odsParser.getContext(),
      ::mlir::Type((_result_type.value_or(::mlir::NoneType::get(odsParser.getContext())))),
      llvm::SmallVector<primitive::BoolAttr>((*_result_value)));
}

void BoolArrAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << '[';
  for (int i = 0; i < getValue().size(); ++i){
	  odsPrinter << getValue()[i].getValueStr() ;
	  if (i < getValue().size()-1){
		odsPrinter << ',';
	  }
  }
  odsPrinter << ']';
  //odsPrinter.printStrippedAttrOrType(getValue());
}

} //namespace mlir::toylang::arrays

