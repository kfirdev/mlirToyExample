#include "include/ToyLang/Dialect/Primitive/PrimitiveDialect.h"
#include "include/ToyLang/Dialect/Arrays/ArraysDialect.h"
#include "include/ToyLang/Conversions/Primitive/PrimitiveToStandard.h"
#include "include/ToyLang/Passes/Primitive/Passes.h"
#include "include/ToyLang/Passes/Primitive/PrintPass.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"

void primitiveToLLVMPipelineBuilder(mlir::OpPassManager &manager){
	manager.addPass(mlir::toylang::primitive::createPrimToStandard());
	manager.addPass(mlir::createCanonicalizerPass());
	manager.addPass(mlir::createConvertFuncToLLVMPass());
	manager.addPass(mlir::createArithToLLVMConversionPass());
}

int main(int argc, char **argv) {
	mlir::DialectRegistry registry;
	registry.insert<mlir::toylang::primitive::PrimitiveDialect>();
	registry.insert<mlir::toylang::arrays::ArraysDialect>();
	mlir::registerAllDialects(registry);
	mlir::registerAllPasses();
	mlir::toylang::primitive::passes::registerPrintPass();
	mlir::toylang::primitive::registerPrimToStandardPass();
	
	mlir::PassPipelineRegistration<>("primitive-to-llvm",
			"Run passes to lower primitive dialect to llvm",
			primitiveToLLVMPipelineBuilder);

	return mlir::asMainReturnCode(
  	    mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}

