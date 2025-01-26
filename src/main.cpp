#include "include/ToyLang/Dialect/Primitive/PrimitiveDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "ToyLang/Passes/Primitive/Passes.h"
#include "ToyLang/Passes/Primitive/PrintPass.h"
//#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
	mlir::DialectRegistry registry;
	registry.insert<mlir::toylang::primitive::PrimitiveDialect>();
	mlir::registerAllDialects(registry);
	mlir::registerAllPasses();
	//mlir::registerPass
	mlir::toylang::primitive::registerPasses();
	mlir::toylang::primitive::registerPrintPassPass();
	mlir::toylang::primitive::registerPrintPass();
	//mlir::PassRegistration<mlir::toylang::primitive::Print

	//mlir::PassRegistration<mlir::toylang::primitive::Print

	return mlir::asMainReturnCode(
  	    mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}

