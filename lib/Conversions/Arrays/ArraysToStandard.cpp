#include "include/ToyLang/Conversions/Arrays/ArraysToStandard.h"
#include "ToyLang/Dialect/Primitive/PrimitiveOps.h"
#include "include/ToyLang/Conversions/Primitive/PrimitiveToStandard.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"
#include "include/ToyLang/Dialect/Arrays/ArraysDialect.h"
#include "include/ToyLang/Dialect/Arrays/ArraysType.h"
#include "include/ToyLang/Dialect/Arrays/ArraysOps.h"
#include "include/ToyLang/Dialect/Arrays/ArraysAttr.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::toylang::arrays{

#define GEN_PASS_DEF_ARRTOSTANDARD 
#include "ToyLang/Conversions/Arrays/ArraysToStandard.h.inc"

class ArraysToStandardTypeConverter : public TypeConverter {
	public:
		ArraysToStandardTypeConverter(MLIRContext *ctx) {
			addConversion([](Type type) { return type; });
 	 	  	addConversion([ctx](ArrayType type) -> Type {
					auto convSubType = type.getType().toStandard();
					return mlir::RankedTensorType::get(type.getLength(),convSubType);
 	 	  	});
			addSourceMaterialization([](OpBuilder &builder, Type type,
    		                            ValueRange inputs, Location loc) -> Value {
    		  return builder.create<primitive::FromStandardOp>(loc, type, inputs[0]);
    		});
 	 	}
};

struct ConvertInsert : public mlir::OpConversionPattern<InsertOp>{
	ConvertInsert(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<InsertOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(InsertOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {

		auto convertedScalar = rewriter.create<primitive::ToStandardOp>(
				op.getLoc(), op.getScalar().getType().toStandard(), adaptor.getScalar());

		tensor::InsertOp insertOp = rewriter.create<tensor::InsertOp>(
				op.getLoc(),convertedScalar.getResult(),adaptor.getDest(),adaptor.getIndices());

		rewriter.replaceOp(op.getOperation(), insertOp.getOperation());
		return llvm::success();
	}
};

struct ConvertConcat : public mlir::OpConversionPattern<ConcatOp>{
	ConvertConcat(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<ConcatOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(ConcatOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {

		tensor::ConcatOp concatOp = rewriter.create<tensor::ConcatOp>(op.getLoc(),0,adaptor.getOperands());

		rewriter.replaceOp(op.getOperation(), concatOp.getOperation());
		return llvm::success();
	}
};

struct ConvertExtract : public mlir::OpConversionPattern<ExtractOp>{
	ConvertExtract(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<ExtractOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(ExtractOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {

		tensor::ExtractOp extractOp = rewriter.create<tensor::ExtractOp>(op.getLoc(),adaptor.getTensor(),adaptor.getIndices());

		rewriter.replaceOp(op.getOperation(), extractOp.getOperation());
		return llvm::success();
	}
};

struct ConvertConstant : public mlir::OpConversionPattern<ConstantOp>{
	ConvertConstant(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<ConstantOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(ConstantOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {

		mlir::Type intType = op.getResult().getType().getType().toStandard();
		mlir::RankedTensorType type = mlir::RankedTensorType::get(op.getResult().getType().getLength(),intType);
		llvm::SmallVector<mlir::Attribute> arr;
		arr.reserve(op.getValue().getValue().size());
		for (primitive::PrimitiveAttrInterface attr : op.getValue().getValue()){
			arr.push_back(attr.toStandard());
		}
		mlir::DenseElementsAttr intAttr = mlir::DenseElementsAttr::get(type,arr);
		arith::ConstantOp constOp = rewriter.create<arith::ConstantOp>(op.getLoc(),intAttr);
		rewriter.replaceOp(op.getOperation(), constOp.getOperation());
		return llvm::success();
	}
};

struct ArrToStandard : impl::ArrToStandardBase<ArrToStandard> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

	ConversionTarget target(*context);
	target.addIllegalDialect<ArraysDialect>();
	target.addLegalDialect<arith::ArithDialect>();
	target.addLegalDialect<tensor::TensorDialect>();
	target.addLegalDialect<primitive::PrimitiveDialect>();

	mlir::RewritePatternSet patterns(context);
	ArraysToStandardTypeConverter type_convertor(context);
	patterns.add<ConvertConstant,ConvertInsert,ConvertConcat,ConvertExtract>(type_convertor,context);

	populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, type_convertor);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return type_convertor.isSignatureLegal(op.getFunctionType()) &&
             type_convertor.isLegal(&op.getBody());
    });

    populateReturnOpTypeConversionPattern(patterns, type_convertor);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return type_convertor.isLegal(op); });

    populateCallOpTypeConversionPattern(patterns, type_convertor);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return type_convertor.isLegal(op); });

    populateBranchOpInterfaceTypeConversionPattern(patterns, type_convertor);
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                              type_convertor) ||
             isLegalForReturnOpTypeConversionPattern(op, type_convertor);
    });

	if (failed(applyPartialConversion(module, target, std::move(patterns)))){
		signalPassFailure();
	}
  }
};

}
