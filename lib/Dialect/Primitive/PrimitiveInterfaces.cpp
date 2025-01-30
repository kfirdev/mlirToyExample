#include "include/ToyLang/Dialect/Primitive/PrimitiveAttr.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveInterfaces.h"

namespace mlir::toylang::primitive{

// IntegerAttr
PrimitiveAttrInterface IntegerAttr::add(PrimitiveAttrInterface& other) const{
	auto intAttr = mlir::cast<IntegerAttr>(other);
	return IntegerAttr::get(getType(),getValue()+intAttr.getValue());
}
PrimitiveAttrInterface IntegerAttr::sub(PrimitiveAttrInterface& other) const{
	auto intAttr = mlir::cast<IntegerAttr>(other);
	return IntegerAttr::get(getType(),getValue()-intAttr.getValue());
}
PrimitiveAttrInterface IntegerAttr::mult(PrimitiveAttrInterface& other) const{
	auto intAttr = mlir::cast<IntegerAttr>(other);
	return IntegerAttr::get(getType(),getValue()*intAttr.getValue());
}
PrimitiveAttrInterface IntegerAttr::div(PrimitiveAttrInterface& other) const{
	auto intAttr = mlir::cast<IntegerAttr>(other);
	return IntegerAttr::get(getType(),getValue().sdiv(intAttr.getValue()));
}
std::string IntegerAttr::getValueStr() const{
	std::string valueStr;
    llvm::raw_string_ostream valueStream(valueStr);
	getValue().print(valueStream, true);
    valueStream.flush();
	return valueStr;
}
unsigned IntegerAttr::getWidth() const{
	return getValue().getBitWidth();
}
unsigned IntegerAttr::getActiveWidth() const{
	return getValue().getActiveBits();
}

} // namespace mlir::toylang::primitive
