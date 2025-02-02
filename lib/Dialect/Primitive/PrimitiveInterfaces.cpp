#include "include/ToyLang/Dialect/Primitive/PrimitiveAttr.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveInterfaces.h"

namespace mlir::toylang::primitive{
//IntegerType
mlir::Type IntegerType::toStandard() const{
	return mlir::IntegerType::get(getContext(),getWidth(),mlir::IntegerType::Signless);
}

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
mlir::Operation* IntegerAttr::toStandard(ConversionPatternRewriter& rewriter,mlir::Location loc) const{

	mlir::IntegerType intType = mlir::IntegerType::get(getContext(), getWidth());
	mlir::IntegerAttr intAttr = mlir::IntegerAttr::get(intType, getValue());

	
	return rewriter.create<arith::ConstantOp>(loc,intAttr);
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

//FloatType
mlir::Type FloatType::toStandard() const{
	//return mlir::FloatType::get(getContext(),getWidth(),mlir::Float32Type::Signless);
	return mlir::Float32Type{};
}

//FloatAttr

PrimitiveAttrInterface FloatAttr::add(PrimitiveAttrInterface& other) const{
	auto intAttr = mlir::cast<FloatAttr>(other);
	return FloatAttr::get(getType(),getValue()+intAttr.getValue());
}
PrimitiveAttrInterface FloatAttr::sub(PrimitiveAttrInterface& other) const{
	auto intAttr = mlir::cast<FloatAttr>(other);
	return FloatAttr::get(getType(),getValue()-intAttr.getValue());
}
PrimitiveAttrInterface FloatAttr::mult(PrimitiveAttrInterface& other) const{
	auto intAttr = mlir::cast<FloatAttr>(other);
	return FloatAttr::get(getType(),getValue()*intAttr.getValue());
}
PrimitiveAttrInterface FloatAttr::div(PrimitiveAttrInterface& other) const{
	auto floatAttr = mlir::cast<FloatAttr>(other);
	llvm::APFloat result = getValue();
	auto _ = result.divide(floatAttr.getValue(),llvm::RoundingMode::Dynamic);
	return FloatAttr::get(getType(),result);
}
mlir::Operation* FloatAttr::toStandard(ConversionPatternRewriter& rewriter,mlir::Location loc) const{

	//mlir::FloatType intType = mlir::Float32Type;
	mlir::FloatAttr intAttr = mlir::FloatAttr::get(mlir::Float32Type{}, getValue());

	
	return rewriter.create<arith::ConstantOp>(loc,intAttr);
}
std::string FloatAttr::getValueStr() const{
	std::string valueStr;
    llvm::raw_string_ostream valueStream(valueStr);
	getValue().print(valueStream);
    valueStream.flush();
	return valueStr;
}
unsigned FloatAttr::getWidth() const{
	//	return getValue().getBitWidth();
	const auto &sem = getValue().getSemantics();
	const auto &sem16 = llvm::APFloat::IEEEhalf();
	const auto &sem32 = llvm::APFloat::IEEEsingle();
	const auto &sem64 = llvm::APFloat::IEEEdouble();
	if (&sem == &sem16)
		return 16;
	if (&sem == &sem32)
		return 32;
	if (&sem == &sem64)
		return 64;
	return 0;
}
unsigned FloatAttr::getActiveWidth() const{
	//return getValue().getActiveBits();
	return getWidth();
}

} // namespace mlir::toylang::primitive
