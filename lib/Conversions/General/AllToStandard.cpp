#include "ToyLang/Dialect/Primitive/PrimitiveOps.h"
#include "include/ToyLang/Conversions/General/AllToStandard.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"
#include "include/ToyLang/Dialect/Arrays/ArraysDialect.h"
#include "include/ToyLang/Dialect/Arrays/ArraysType.h"
#include "include/ToyLang/Dialect/Arrays/ArraysOps.h"
#include "include/ToyLang/Dialect/Arrays/ArraysAttr.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::toylang{

#define GEN_PASS_DEF_ALLTOSTANDARD 
#include "ToyLang/Conversions/General/AllToStandard.h.inc"

class AllToStandardTypeConverter : public TypeConverter {
	public:
		AllToStandardTypeConverter(MLIRContext *ctx) {
			addConversion([](Type type) { return type; });
 	 	  	addConversion([ctx](primitive::PrimitiveTypeInterface type) -> Type {
				return type.toStandard();
 	 	  	});
 	 	  	addConversion([ctx](arrays::ArrayType type) -> Type {
					auto convSubType = type.getType().toStandard();
					if (type.getLength() != 0){
						return mlir::RankedTensorType::get(type.getLength(),convSubType);
					}
					else{
						return mlir::RankedTensorType::get(mlir::ShapedType::kDynamic,convSubType);
					}
 	 	  	});
			//addSourceMaterialization([](OpBuilder &builder, Type type,
    		//                            ValueRange inputs, Location loc) -> Value {
    		//  return builder.create<primitive::FromStandardOp>(loc, type, inputs[0]);
    		//});
 	 	}
};

struct ConvertInsert : public mlir::OpConversionPattern<arrays::InsertOp>{
	ConvertInsert(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<arrays::InsertOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(arrays::InsertOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {

		auto convertedScalar = rewriter.create<primitive::ToStandardOp>(
				op.getLoc(), op.getScalar().getType().toStandard(), adaptor.getScalar());

		tensor::InsertOp insertOp = rewriter.create<tensor::InsertOp>(
				op.getLoc(),convertedScalar.getResult(),adaptor.getDest(),adaptor.getIndices());

		rewriter.replaceOp(op.getOperation(), insertOp.getOperation());
		return llvm::success();
	}
};

struct ConvertConcat : public mlir::OpConversionPattern<arrays::ConcatOp>{
	ConvertConcat(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<arrays::ConcatOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(arrays::ConcatOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {

		tosa::ConcatOp concatOp = rewriter.create<tosa::ConcatOp>(op.getLoc(),adaptor.getOperands(),0);

		rewriter.replaceOp(op.getOperation(), concatOp.getOperation());
		return llvm::success();
	}
};

struct ConvertExtract : public mlir::OpConversionPattern<arrays::ExtractOp>{
	ConvertExtract(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<arrays::ExtractOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(arrays::ExtractOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {

		tensor::ExtractOp extractOp = rewriter.create<tensor::ExtractOp>(op.getLoc(),adaptor.getTensor(),adaptor.getIndices());

		rewriter.replaceOp(op.getOperation(), extractOp.getOperation());
		return llvm::success();
	}
};
struct ConvertCast : public mlir::OpConversionPattern<arrays::CastOp>{
	ConvertCast(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<arrays::CastOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(arrays::CastOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {

		tensor::CastOp castOp = rewriter.create<tensor::CastOp>(
				op.getLoc(),getTypeConverter()->convertType(op.getDest().getType()),adaptor.getSource());

		rewriter.replaceOp(op.getOperation(), castOp.getOperation());
		return llvm::success();
	}
};

struct ConvertConstant : public mlir::OpConversionPattern<arrays::ConstantOp>{
	ConvertConstant(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<arrays::ConstantOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(arrays::ConstantOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {

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

struct ConvertAdd : public mlir::OpConversionPattern<primitive::AddOp>{
	ConvertAdd(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<primitive::AddOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(primitive::AddOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		mlir::Operation* addOp = op.getType().addToStandard(rewriter,op.getLoc(),adaptor.getLhs(),adaptor.getRhs());
		rewriter.replaceOp(op.getOperation(), addOp);
		return llvm::success();
	}
};
struct ConvertSub : public mlir::OpConversionPattern<primitive::SubOp>{
	ConvertSub(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<primitive::SubOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(primitive::SubOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		mlir::Operation* subOp = op.getType().subToStandard(rewriter,op.getLoc(),adaptor.getLhs(),adaptor.getRhs());
		rewriter.replaceOp(op.getOperation(), subOp);
		return llvm::success();
	}
};

struct ConvertMult : public mlir::OpConversionPattern<primitive::MultOp>{
	ConvertMult(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<primitive::MultOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(primitive::MultOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		mlir::Operation* multOp = op.getType().multToStandard(rewriter,op.getLoc(),adaptor.getLhs(),adaptor.getRhs());
		rewriter.replaceOp(op.getOperation(), multOp);
		return llvm::success();
	}
};

struct ConvertDiv : public mlir::OpConversionPattern<primitive::DivOp>{
	ConvertDiv(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<primitive::DivOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(primitive::DivOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		mlir::Operation* divOp = op.getType().divToStandard(rewriter,op.getLoc(),adaptor.getLhs(),adaptor.getRhs());
		rewriter.replaceOp(op.getOperation(), divOp);
		return llvm::success();
	}
};

struct ConvertIf : public mlir::OpConversionPattern<primitive::IfOp>{
	ConvertIf(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<primitive::IfOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(primitive::IfOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {

		mlir::Type resType = mlir::cast<primitive::PrimitiveTypeInterface>(op.getResult().getType().front()).toStandard();
		scf::IfOp ifOp = rewriter.create<scf::IfOp>(
				op.getLoc(),resType,adaptor.getCondition(),
				!adaptor.getThenRegion().empty(),!adaptor.getElseRegion().empty());


		rewriter.eraseBlock(&ifOp.getThenRegion().front());
		// Inline the 'then' region from the original op into the new scf.if.
    	// This moves the single block from op.getThenRegion() into newIf.thenRegion().
    	rewriter.inlineRegionBefore(op.getThenRegion(), ifOp.getThenRegion(),
    	                            ifOp.getThenRegion().begin());

    	// If there is an else region, inline it similarly.
    	if (!adaptor.getElseRegion().empty()) {
		  rewriter.eraseBlock(&ifOp.getElseRegion().front());
    	  rewriter.inlineRegionBefore(op.getElseRegion(), ifOp.getElseRegion(),
    	                              ifOp.getElseRegion().begin());
    	}
    	
    	// Replace the original op with the results of the new scf.if.
    	rewriter.replaceOp(op.getOperation(), ifOp.getOperation());
    	return success();
	}
};

struct ConvertFor : public mlir::OpConversionPattern<primitive::ForOp>{
	ConvertFor(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<primitive::ForOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(primitive::ForOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		scf::ForOp forOp = rewriter.create<scf::ForOp>(op.getLoc(),
				adaptor.getLowerBound(),adaptor.getHigherBound(),adaptor.getStep(),
				adaptor.getInitArgs());

		if (failed(rewriter.convertRegionTypes(&op.getRegion(), *getTypeConverter())))
			return failure();

		rewriter.eraseBlock(&forOp.getRegion().front());
    	rewriter.inlineRegionBefore(op.getRegion(), forOp.getRegion(),
    	                            forOp.getRegion().begin());


		rewriter.replaceOp(op.getOperation(),forOp.getOperation());
		return success();
	}
};


struct ConvertYield : public mlir::OpConversionPattern<primitive::YieldOp>{
	ConvertYield(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<primitive::YieldOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(primitive::YieldOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		mlir::scf::YieldOp yieldOp = rewriter.create<scf::YieldOp>(op.getLoc(),adaptor.getResults());
		rewriter.replaceOp(op.getOperation(), yieldOp.getOperation());
		return llvm::success();
	}
};

struct ConvertFunc : public mlir::OpConversionPattern<primitive::FuncOp>{
	ConvertFunc(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<primitive::FuncOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(primitive::FuncOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                    op.getFunctionType());
		rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
		rewriter.eraseOp(op);
		return llvm::success();
	}
};

struct ConvertReturn : public mlir::OpConversionPattern<primitive::ReturnOp>{
	ConvertReturn(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<primitive::ReturnOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(primitive::ReturnOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		func::ReturnOp returnOp = rewriter.create<func::ReturnOp>(op.getLoc(),adaptor.getOperands());
		rewriter.replaceOp(op.getOperation(), returnOp.getOperation());
		return llvm::success();
	}
};
struct ConvertGenericCall : public mlir::OpConversionPattern<primitive::GenericCallOp>{
	ConvertGenericCall(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<primitive::GenericCallOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(primitive::GenericCallOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		mlir::func::CallOp callOp = rewriter.create<mlir::func::CallOp>(
				op.getLoc(), op.getCallee(), 
				op.getResult().getType(), op.getOperands());
		rewriter.replaceOp(op.getOperation(),callOp.getOperation());
		return llvm::success();
	}
};
struct ConvertPrimConstant : public mlir::OpConversionPattern<primitive::ConstantOp>{
	ConvertPrimConstant(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<primitive::ConstantOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(primitive::ConstantOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		mlir::TypedAttr Attr = op.getValue().toStandard();
		arith::ConstantOp constOp = rewriter.create<arith::ConstantOp>(op.getLoc(),Attr);
		rewriter.replaceOp(op.getOperation(), constOp.getOperation());
		return llvm::success();
	}
};

struct ConvertToStandard : public mlir::OpConversionPattern<primitive::ToStandardOp>{
	ConvertToStandard(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<primitive::ToStandardOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(primitive::ToStandardOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		rewriter.replaceOp(op.getOperation(),op.getOperand());
		return llvm::success();
	}
};

struct ConvertFromStandard : public mlir::OpConversionPattern<primitive::FromStandardOp>{
	ConvertFromStandard(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<primitive::FromStandardOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(primitive::FromStandardOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		rewriter.replaceOp(op.getOperation(),op.getOperand());
		return llvm::success();
	}
};
struct AllToStandard : impl::AllToStandardBase<AllToStandard> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

	ConversionTarget target(*context);
	target.addIllegalDialect<arrays::ArraysDialect>();
	target.addIllegalDialect<primitive::PrimitiveDialect>();

	target.addLegalDialect<arith::ArithDialect>();
	target.addLegalDialect<tensor::TensorDialect>();
	target.addLegalDialect<scf::SCFDialect>();

	mlir::RewritePatternSet patterns(context);
	AllToStandardTypeConverter type_convertor(context);
	patterns.add<ConvertConstant,ConvertInsert,ConvertConcat,ConvertExtract,ConvertCast>(type_convertor,context);
	patterns.add<
	ConvertAdd,
	ConvertSub,
	ConvertMult,
	ConvertDiv,
	ConvertIf,
	ConvertFor,
    ConvertYield,
	ConvertFunc,
	ConvertGenericCall,
	ConvertReturn,
	ConvertToStandard,
	ConvertFromStandard,
	ConvertPrimConstant
	>(type_convertor,context);

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
} //namespace mlir::toylang
