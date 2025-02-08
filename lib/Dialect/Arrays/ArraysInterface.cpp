#include "include/ToyLang/Dialect/Arrays/ArraysInterface.h"
#include "ToyLang/Dialect/Arrays/ArraysType.h"

namespace mlir::toylang::arrays{
//===----------------------------------------------------------------------===//
// BoolArrType
//===----------------------------------------------------------------------===//
	unsigned BoolArrType::getWidth() const{
		return 1;
	}
	mlir::Type BoolArrType::getCombined(unsigned other_length) const{
		return BoolArrType::get(getContext(), other_length+getLength());
	}
//===----------------------------------------------------------------------===//
// IntegerArrType
//===----------------------------------------------------------------------===//
mlir::Type IntegerArrType::getCombined(unsigned other_length) const{
	return IntegerArrType::get(getContext(), other_length+getLength(),getWidth());
}
//===----------------------------------------------------------------------===//
// FloatArrType
//===----------------------------------------------------------------------===//
mlir::Type FloatArrType::getCombined(unsigned other_length) const{
	return FloatArrType::get(getContext(), other_length+getLength(),getWidth());
}
	

}

