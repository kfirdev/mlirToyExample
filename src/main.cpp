#include "include/ToyLang/Dialect/ToyDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
	mlir::DialectRegistry registry;
	registry.insert<mlir::toylang::toy::ToyDialect>();
	mlir::registerAllDialects(registry);
	mlir::registerAllDialects(registry);

	return mlir::asMainReturnCode(
  	    mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}

