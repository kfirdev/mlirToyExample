#include "include/ToyLang/Conversions/Primitive/PrimitiveToStandard.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveDialect.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveAttr.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveTypes.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::toylang::primitive{

#define GEN_PASS_DEF_PRIMTOSTANDARD
#include "ToyLang/Conversions/Primitive/PrimitiveToStandard.h.inc"

class PrimitiveToStandardTypeConverter : public TypeConverter {
	public:
		PrimitiveToStandardTypeConverter(MLIRContext *ctx) {
			addConversion([](Type type) { return type; });
 	 	  	addConversion([ctx](IntegerType type) -> Type {
				return mlir::IntegerType::get(type.getContext(),type.getWidth(),mlir::IntegerType::Signless);
 	 	  	});
 	 	}
};

struct ConvertAdd : public mlir::OpConversionPattern<AddOp>{
	ConvertAdd(mlir::TypeConverter& type_convertor, MLIRContext* context) 
		: mlir::OpConversionPattern<AddOp>(type_convertor,context){}

	LogicalResult matchAndRewrite(AddOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
		arith::AddIOp addOp = rewriter.create<arith::AddIOp>(
				op.getLoc(), adaptor.getLhs(), adaptor.getRhs());

		rewriter.replaceOp(op.getOperation(), addOp.getOperation());
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
