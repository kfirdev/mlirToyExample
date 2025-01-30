#include "ToyLang/Passes/Primitive/PrintPass.h"
#include "ToyLang/Dialect/Primitive/PrimitiveDialect.h"
#include "ToyLang/Dialect/Primitive/PrimitiveAttr.h"
#include "ToyLang/Dialect/Primitive/PrimitiveOps.h"


namespace mlir::toylang::primitive{


#define GEN_PASS_DEF_PRINTPASS
#include "ToyLang/Passes/Primitive/PrintPass.h.inc"
struct PrintPass : impl::PrintPassBase<PrintPass>{
	void runOnOperation() override{
		  getOperation()->walk([&](ConstantOp op) {
		llvm::outs() << "ConstantOp found with bit width: " << mlir::cast<IntegerAttr>(op.getValue()).getValue().getBitWidth() << "\n";
		llvm::outs() << "Should be with bit width: " << op.getResult().getType().getWidth() << "\n";
		//op.getOutput().getType().get
	});
	}
};

} // namespace
