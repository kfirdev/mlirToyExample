#include "include/ToyLang/Dialect/Primitive/PrimitiveDialect.h"
#include "include/ToyLang/Conversions/Primitive/PrimitiveToStandard.h"
#include "include/ToyLang/Passes/Primitive/Passes.h"
#include "include/ToyLang/Passes/Primitive/PrintPass.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
//#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
	mlir::DialectRegistry registry;
	registry.insert<mlir::toylang::primitive::PrimitiveDialect>();
	mlir::registerAllDialects(registry);
	mlir::registerAllPasses();
	mlir::toylang::primitive::passes::registerPrintPass();
	mlir::toylang::primitive::registerPrimToStandardPass();

	return mlir::asMainReturnCode(
  	    mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}

