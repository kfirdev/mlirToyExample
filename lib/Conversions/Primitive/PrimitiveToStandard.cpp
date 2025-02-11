#include "include/ToyLang/Conversions/Primitive/PrimitiveToStandard.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveDialect.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveTypes.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveOps.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveAttr.h"

namespace mlir::toylang::primitive{

#define GEN_PASS_DEF_PRIMTOSTANDARD
#include "ToyLang/Conversions/Primitive/PrimitiveToStandard.h.inc"

class PrimitiveToStandardTypeConverter : public TypeConverter {
	public:
		PrimitiveToStandardTypeConverter(MLIRContext *ctx) {
			addConversion([](Type type) { return type; });
 	 	  	addConversion([ctx](PrimitiveTypeInterface type) -> Type {
				return type.toStandard();
 	 	  	});
 	 	}
};

struct ConvertAdd : public mlir::OpConversionPattern<AddOp>{
	ConvertAdd(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<AddOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(AddOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		mlir::Operation* addOp = op.getType().addToStandard(rewriter,op.getLoc(),adaptor.getLhs(),adaptor.getRhs());
		rewriter.replaceOp(op.getOperation(), addOp);
		return llvm::success();
	}
};
struct ConvertSub : public mlir::OpConversionPattern<SubOp>{
	ConvertSub(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<SubOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(SubOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		mlir::Operation* subOp = op.getType().subToStandard(rewriter,op.getLoc(),adaptor.getLhs(),adaptor.getRhs());
		rewriter.replaceOp(op.getOperation(), subOp);
		return llvm::success();
	}
};

struct ConvertMult : public mlir::OpConversionPattern<MultOp>{
	ConvertMult(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<MultOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(MultOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		mlir::Operation* multOp = op.getType().multToStandard(rewriter,op.getLoc(),adaptor.getLhs(),adaptor.getRhs());
		rewriter.replaceOp(op.getOperation(), multOp);
		return llvm::success();
	}
};

struct ConvertDiv : public mlir::OpConversionPattern<DivOp>{
	ConvertDiv(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<DivOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(DivOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		mlir::Operation* divOp = op.getType().divToStandard(rewriter,op.getLoc(),adaptor.getLhs(),adaptor.getRhs());
		rewriter.replaceOp(op.getOperation(), divOp);
		return llvm::success();
	}
};

struct ConvertIf : public mlir::OpConversionPattern<IfOp>{
	ConvertIf(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<IfOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(IfOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {

		mlir::Type resType = mlir::cast<PrimitiveTypeInterface>(op.getResult().getType().front()).toStandard();
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
struct ConvertYield : public mlir::OpConversionPattern<YieldOp>{
	ConvertYield(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<YieldOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(YieldOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		mlir::scf::YieldOp yieldOp = rewriter.create<scf::YieldOp>(op.getLoc(),adaptor.getResults());
		rewriter.replaceOp(op.getOperation(), yieldOp.getOperation());
		return llvm::success();
	}
};

struct ConvertConstant : public mlir::OpConversionPattern<ConstantOp>{
	ConvertConstant(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<ConstantOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(ConstantOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		mlir::TypedAttr Attr = op.getValue().toStandard();
		arith::ConstantOp constOp = rewriter.create<arith::ConstantOp>(op.getLoc(),Attr);
		rewriter.replaceOp(op.getOperation(), constOp.getOperation());
		return llvm::success();
	}
};

struct ConvertToStandard : public mlir::OpConversionPattern<ToStandardOp>{
	ConvertToStandard(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<ToStandardOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(ToStandardOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		rewriter.replaceOp(op.getOperation(),op.getOperand());
		return llvm::success();
	}
};

struct ConvertFromStandard : public mlir::OpConversionPattern<FromStandardOp>{
	ConvertFromStandard(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<FromStandardOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(FromStandardOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		rewriter.replaceOp(op.getOperation(),op.getOperand());
		return llvm::success();
	}
};

struct PrimToStandard : impl::PrimToStandardBase<PrimToStandard> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

	ConversionTarget target(*context);
	target.addIllegalDialect<PrimitiveDialect>();
	target.addLegalDialect<arith::ArithDialect>();
	target.addLegalDialect<scf::SCFDialect>();

	mlir::RewritePatternSet patterns(context);
	PrimitiveToStandardTypeConverter type_convertor(context);
	patterns.add<
	ConvertAdd,
	ConvertSub,
	ConvertMult,
	ConvertDiv,
	ConvertIf,
    ConvertYield,
	ConvertToStandard,
	ConvertFromStandard,
	ConvertConstant
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

}
