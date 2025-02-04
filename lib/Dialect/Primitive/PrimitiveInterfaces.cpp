#include "include/ToyLang/Dialect/Primitive/PrimitiveAttr.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/SmallVector.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveInterfaces.h"

namespace mlir::toylang::primitive{
//IntegerType
mlir::Type IntegerType::toStandard() const{
	return mlir::IntegerType::get(getContext(),getWidth(),mlir::IntegerType::Signless);
}
mlir::Operation* IntegerType::addToStandard(ConversionPatternRewriter& rewriter,mlir::Location loc,mlir::Value lhs, mlir::Value rhs){
	return rewriter.create<arith::AddIOp>(
			loc, lhs, rhs).getOperation();
}
mlir::Operation* IntegerType::subToStandard(ConversionPatternRewriter& rewriter,mlir::Location loc,mlir::Value lhs, mlir::Value rhs){
	return rewriter.create<arith::SubIOp>(
			loc, lhs, rhs).getOperation();
}
mlir::Operation* IntegerType::divToStandard(ConversionPatternRewriter& rewriter,mlir::Location loc,mlir::Value lhs, mlir::Value rhs){
	return rewriter.create<arith::DivSIOp>(
			loc, lhs, rhs).getOperation();
}
mlir::Operation* IntegerType::multToStandard(ConversionPatternRewriter& rewriter,mlir::Location loc,mlir::Value lhs, mlir::Value rhs){
	return rewriter.create<arith::MulIOp>(
			loc, lhs, rhs).getOperation();
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
	switch(getWidth()){
		case 16:
			return mlir::Float16Type::get(getContext());
		case 32:
			return mlir::Float32Type::get(getContext());
		case 64:
			return mlir::Float64Type::get(getContext());
		default:
			return mlir::Float64Type::get(getContext());
	}
}
mlir::Operation* FloatType::addToStandard(ConversionPatternRewriter& rewriter,mlir::Location loc,mlir::Value lhs, mlir::Value rhs){
	return rewriter.create<arith::AddFOp>(
			loc, lhs, rhs).getOperation();
}
mlir::Operation* FloatType::subToStandard(ConversionPatternRewriter& rewriter,mlir::Location loc,mlir::Value lhs, mlir::Value rhs){
	return rewriter.create<arith::SubFOp>(
			loc, lhs, rhs).getOperation();
}
mlir::Operation* FloatType::divToStandard(ConversionPatternRewriter& rewriter,mlir::Location loc,mlir::Value lhs, mlir::Value rhs){
	return rewriter.create<arith::DivFOp>(
			loc, lhs, rhs).getOperation();
}
mlir::Operation* FloatType::multToStandard(ConversionPatternRewriter& rewriter,mlir::Location loc,mlir::Value lhs, mlir::Value rhs){
	return rewriter.create<arith::MulFOp>(
			loc, lhs, rhs).getOperation();
}

//FloatAttr
PrimitiveAttrInterface FloatAttr::add(PrimitiveAttrInterface& other) const{
	auto intAttr = mlir::cast<FloatAttr>(other);
	llvm::APFloat result = getValue();
	result.add(intAttr.getValue(),llvm::RoundingMode::NearestTiesToAway);
	return FloatAttr::get(getType(),result);
}
PrimitiveAttrInterface FloatAttr::sub(PrimitiveAttrInterface& other) const{
	auto intAttr = mlir::cast<FloatAttr>(other);
	llvm::APFloat result = getValue();
	result.subtract(intAttr.getValue(),llvm::RoundingMode::NearestTiesToAway);
	return FloatAttr::get(getType(),result);
}
PrimitiveAttrInterface FloatAttr::mult(PrimitiveAttrInterface& other) const{
	auto intAttr = mlir::cast<FloatAttr>(other);
	llvm::APFloat result = getValue();
	result.multiply(intAttr.getValue(),llvm::RoundingMode::NearestTiesToAway);
	return FloatAttr::get(getType(),result);
}
PrimitiveAttrInterface FloatAttr::div(PrimitiveAttrInterface& other) const{
	auto floatAttr = mlir::cast<FloatAttr>(other);
	llvm::APFloat result = getValue();
	auto _ = result.divide(floatAttr.getValue(),llvm::RoundingMode::NearestTiesToAway);
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

//BoolAttr
PrimitiveAttrInterface BoolAttr::add(PrimitiveAttrInterface& other) const{
	return nullptr;
}
PrimitiveAttrInterface BoolAttr::sub(PrimitiveAttrInterface& other) const{
	return nullptr;
}
PrimitiveAttrInterface BoolAttr::div(PrimitiveAttrInterface& other) const{
	auto intAttr = mlir::cast<BoolAttr>(other);
	return BoolAttr::get(getType(),getValue() || intAttr.getValue());
}
PrimitiveAttrInterface BoolAttr::mult(PrimitiveAttrInterface& other) const{
	auto intAttr = mlir::cast<BoolAttr>(other);
	return BoolAttr::get(getType(),getValue() && intAttr.getValue());
}
mlir::Operation* BoolAttr::toStandard(ConversionPatternRewriter& rewriter,mlir::Location loc) const{
	mlir::BoolAttr intAttr = mlir::BoolAttr::get(getContext(), getValue());
	return rewriter.create<arith::ConstantOp>(loc,intAttr);
}
std::string BoolAttr::getValueStr() const{
	return getValue() ? "true" : "false";
}
unsigned BoolAttr::getWidth() const{
	return 1;
}
unsigned BoolAttr::getActiveWidth() const{
	return 1;
}

// BoolType
mlir::Type BoolType::toStandard() const{
	return mlir::IntegerType::get(getContext(),1,mlir::IntegerType::Signless);
}
mlir::Operation* BoolType::addToStandard(ConversionPatternRewriter& rewriter,mlir::Location loc,mlir::Value lhs, mlir::Value rhs){
	//return rewriter.create<arith::AddIOp>(
	//		loc, lhs, rhs).getOperation();
	return nullptr;
}
mlir::Operation* BoolType::subToStandard(ConversionPatternRewriter& rewriter,mlir::Location loc,mlir::Value lhs, mlir::Value rhs){
	//return rewriter.create<arith::SubIOp>(
	//		loc, lhs, rhs).getOperation();
	return nullptr;
}
mlir::Operation* BoolType::divToStandard(ConversionPatternRewriter& rewriter,mlir::Location loc,mlir::Value lhs, mlir::Value rhs){
	return rewriter.create<arith::OrIOp>(
			loc, lhs, rhs).getOperation();
}
mlir::Operation* BoolType::multToStandard(ConversionPatternRewriter& rewriter,mlir::Location loc,mlir::Value lhs, mlir::Value rhs){
	return rewriter.create<arith::AndIOp>(
			loc, lhs, rhs).getOperation();
}
unsigned BoolType::getWidth() const {
	return 1;
}


// StringType
mlir::Type StringType::toStandard() const{
	//return mlir::StringType::get(getContext(),getWidth());
	//return mlir::StringAttr::get(getWidth())
	// TODO: convert to actual type
	return nullptr;
}
mlir::Operation* StringType::addToStandard(ConversionPatternRewriter& rewriter,mlir::Location loc,mlir::Value lhs, mlir::Value rhs){
	// TODO: convert actual op
	return nullptr;
}
mlir::Operation* StringType::subToStandard(ConversionPatternRewriter& rewriter,mlir::Location loc,mlir::Value lhs, mlir::Value rhs){
	return nullptr;
}
mlir::Operation* StringType::divToStandard(ConversionPatternRewriter& rewriter,mlir::Location loc,mlir::Value lhs, mlir::Value rhs){
	return nullptr;
}
mlir::Operation* StringType::multToStandard(ConversionPatternRewriter& rewriter,mlir::Location loc,mlir::Value lhs, mlir::Value rhs){
	return nullptr;
}
unsigned StringType::getWidth() const{
	return 1;
}

// StringAttr
PrimitiveAttrInterface StringAttr::add(PrimitiveAttrInterface& other) const{
	auto intAttr = mlir::cast<StringAttr>(other);
	llvm::SmallVector<IntegerAttr> combined; 
	combined.reserve(getWidth()+other.getWidth());
	for (IntegerAttr val: getValue()){
		combined.push_back(val);
	}
	for (IntegerAttr val: intAttr.getValue()){
		combined.push_back(val);
	}
	return StringAttr::get(getContext(), getType(),combined);
}
PrimitiveAttrInterface StringAttr::sub(PrimitiveAttrInterface& other) const{
	return nullptr;
}
PrimitiveAttrInterface StringAttr::mult(PrimitiveAttrInterface& other) const{
	return nullptr;
}
PrimitiveAttrInterface StringAttr::div(PrimitiveAttrInterface& other) const{
	return nullptr;
}
mlir::Operation* StringAttr::toStandard(ConversionPatternRewriter& rewriter,mlir::Location loc) const{

	//mlir::StringType intType = mlir::StringType::get(getContext(), getWidth());
	//mlir::StringAttr::get();
	
	std::vector<int> res;
	res.reserve(getWidth());
	for (IntegerAttr val: getValue()){
		res.push_back(val.getValue().getZExtValue());
	}
	std::string value {res.begin(),res.end()};

	mlir::StringAttr intAttr = mlir::StringAttr::get(value, getType());
	return rewriter.create<arith::ConstantOp>(loc,intAttr);
}
std::string StringAttr::getValueStr() const{
	std::vector<int> res;
	res.reserve(getWidth());
	for (IntegerAttr val: getValue()){
		res.push_back((int) val.getValue().getZExtValue());
	}
	std::string value {res.begin(),res.end()};
	return value;
}
unsigned StringAttr::getWidth() const{
	return getValue().size();
}
unsigned StringAttr::getActiveWidth() const{
	return getValue().size();
}


} // namespace mlir::toylang::primitive
