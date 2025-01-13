//#include "mlir/IR/MLIRContext.h"
//#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"


//using namespace std::mlir;
using namespace mlir;

int main() {
  // Initialize the MLIR context.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  Builder builder(&context);

  // Create an MLIR module.
  auto module = ModuleOp::create(builder.getUnknownLoc());

  // Create a function type with no inputs and no outputs.
  auto funcType = builder.getFunctionType({}, {});

  // Create an MLIR function inside the module.
  auto func = func::FuncOp::create(builder.getUnknownLoc(), "my_function", funcType);
  func.setPrivate();

  // Add the function to the module.
  module.push_back(func);

  // Print the MLIR module.
  module.print(llvm::outs());

  return 0;
}

