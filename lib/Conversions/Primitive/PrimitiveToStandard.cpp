#include "include/ToyLang/Conversions/Primitive/PrimitiveToStandard.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
		// based on the type convert to different op (has trait on op get type).
		mlir::Operation* addOp;
		if (op.getType().hasTrait<IsAnInteger>()){
			addOp = rewriter.create<arith::AddIOp>(
					op.getLoc(), adaptor.getLhs(), adaptor.getRhs()).getOperation();
		}
		if (op.getType().hasTrait<IsAFloat>()){
			addOp = rewriter.create<arith::AddFOp>(
					op.getLoc(), adaptor.getLhs(), adaptor.getRhs()).getOperation();
		}

		rewriter.replaceOp(op.getOperation(), addOp);
		return llvm::success();
	}
};
struct ConvertSub : public mlir::OpConversionPattern<SubOp>{
	ConvertSub(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<SubOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(SubOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {

		mlir::Operation* subOp;
		if (op.getType().hasTrait<IsAnInteger>()){
			subOp = rewriter.create<arith::SubIOp>(
					op.getLoc(), adaptor.getLhs(), adaptor.getRhs()).getOperation();
		}
		if (op.getType().hasTrait<IsAFloat>()){
			subOp = rewriter.create<arith::SubFOp>(
					op.getLoc(), adaptor.getLhs(), adaptor.getRhs()).getOperation();
		}

		rewriter.replaceOp(op.getOperation(), subOp);
		return llvm::success();
	}
};

struct ConvertMult : public mlir::OpConversionPattern<MultOp>{
	ConvertMult(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<MultOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(MultOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		// based on the type convert to different op (has trait on op get type).
		mlir::Operation* multOp;
		if (op.getType().hasTrait<IsAnInteger>()){
			multOp = rewriter.create<arith::MulIOp>(
					op.getLoc(), adaptor.getLhs(), adaptor.getRhs()).getOperation();
		}
		if (op.getType().hasTrait<IsAFloat>()){
			multOp = rewriter.create<arith::MulFOp>(
					op.getLoc(), adaptor.getLhs(), adaptor.getRhs()).getOperation();
		}

		rewriter.replaceOp(op.getOperation(), multOp);
		return llvm::success();
	}
};
struct ConvertDiv : public mlir::OpConversionPattern<DivOp>{
	ConvertDiv(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<DivOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(DivOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		// based on the type convert to different op (has trait on op get type).
		mlir::Operation* divOp;
		if (op.getType().hasTrait<IsAnInteger>()){
			divOp = rewriter.create<arith::DivSIOp>(
					op.getLoc(), adaptor.getLhs(), adaptor.getRhs()).getOperation();
		}
		if (op.getType().hasTrait<IsAFloat>()){
			divOp = rewriter.create<arith::DivFOp>(
					op.getLoc(), adaptor.getLhs(), adaptor.getRhs()).getOperation();
		}

		rewriter.replaceOp(op.getOperation(), divOp);
		return llvm::success();
	}
};

struct ConvertConstant : public mlir::OpConversionPattern<ConstantOp>{
	ConvertConstant(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<ConstantOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(ConstantOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		mlir::Operation* constOp = op.getValue().toStandard(rewriter,op.getLoc());
		rewriter.replaceOp(op.getOperation(), constOp);
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

	mlir::RewritePatternSet patterns(context);
	PrimitiveToStandardTypeConverter type_convertor(context);
	patterns.add<ConvertAdd>(type_convertor,context);
	patterns.add<ConvertSub>(type_convertor,context);
	patterns.add<ConvertMult>(type_convertor,context);
	patterns.add<ConvertDiv>(type_convertor,context);
	patterns.add<ConvertConstant>(type_convertor,context);

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
